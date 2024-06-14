# coding: utf-8

"""
Selector for triggerstudies
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import, DotDict
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column

from hbw.selection.common import configure_selector, pre_selection, post_selection
from hbw.util import four_vec
from hbw.production.weights import event_weights_to_normalize
from hbw.selection.common import masked_sorted_indices


np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses=(four_vec({"Muon", "Electron", "Jet"})) | {
        "Muon.tightId", "Muon.pfRelIso03_all", "Electron.cutBased", "Electron.mvaIso_WP80",
        "Jet.jetId",
        pre_selection, post_selection,
    },
    produces={"trig_mu_pt", "trig_mu_eta", "trig_ele_pt", "trig_ele_eta", "trig_HT",
              pre_selection, post_selection, 
              "Muon.is_tight", "Electron.is_tight",
              },
    trigger={
        "mu": [
            "Mu12_IsoVVL_PFHT150_PNetBTag_0p53","IsoMu24", "Mu50", 
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55", "Mu15_IsoVVVL_PFHT450",
        ],
        "e": [
            "Ele14_eta2p5_WPTight_Gsf_HT200_PNetBTag_0p53", "Ele28_eta2p1_WPTight_Gsf_HT150", "Ele30_WPTight_Gsf",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55", "Ele15_IsoVVVL_PFHT450",
        ],
    },
    b_tagger= "deepjet",
    btag_wp= "medium",
    exposed=True,
)
def sl_trigger(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    
    events, results = self[pre_selection](events, stats, **kwargs)

    # save triggers
    for channel, trigger_columns in self.config_inst.x.trigger.items():
        for trigger_column in trigger_columns:
            results.steps[f"Trigger_{channel}_{trigger_column}"] = events.HLT[trigger_column]
    
    # leptons
    trig_muon = events.Muon
    trig_electron = events.Electron

    mu_cleaning_mask = (
        (abs(trig_muon.eta) < 2.4) &
        (trig_muon.tightId) &
        (trig_muon.pfRelIso03_all < 0.15)
    )
    ele_cleaning_mask = (
        (abs(trig_electron.eta) < 2.4) &
        (trig_electron.cutBased == 4) &
        (trig_electron.mvaIso_WP80)
    )

    trig_muon_clean = events.Muon[mu_cleaning_mask]
    trig_electron_clean = events.Electron[ele_cleaning_mask]

    events = set_ak_column(events, "trig_mu_pt", ak.fill_none(ak.firsts(trig_muon_clean.pt), -1))
    events = set_ak_column(events, "trig_mu_eta", ak.fill_none(ak.firsts(trig_muon_clean.eta), -100))
    events = set_ak_column(events, "trig_ele_pt", ak.fill_none(ak.firsts(trig_electron_clean.pt), -1))
    events = set_ak_column(events, "trig_ele_eta", ak.fill_none(ak.firsts(trig_electron_clean.eta), -100))

    if self.dataset_inst.is_data == False:

        genparts = events.GenPart
        gen_Ids = genparts.pdgId
        #gen_muon_mask = abs(gen_Ids[events.GenPart.hasFlags("isHardProcess")]) == 13
        #gen_electron_mask = abs(gen_Ids[events.GenPart.hasFlags("isHardProcess")]) == 11

        gen_muon_mask = abs(gen_Ids) == 13
        gen_electron_mask = abs(gen_Ids) == 11

        gen_muons = genparts[gen_muon_mask]
        gen_electrons = genparts[gen_electron_mask]

        muon_mother_is_W = abs(genparts[gen_muons.genPartIdxMother].pdgId) == 24
        electron_mother_is_W = abs(genparts[gen_electrons.genPartIdxMother].pdgId) == 24

        gen_mu_mask = ak.any(muon_mother_is_W, axis=1)
        gen_ele_mask = ak.any(electron_mother_is_W, axis=1)

        results.steps["TrigMuGenMask"] = gen_mu_mask
        results.steps["TrigEleGenMask"] = gen_ele_mask

    mu_presel_mask = ak.sum(trig_muon_clean.pt > 10, axis=1) >= 1
    ele_presel_mask = ak.sum(trig_electron_clean.pt > 10, axis=1) >= 1

    results.steps["TrigMuPreselMask"] = mu_presel_mask
    results.steps["TrigElePreselMask"] = ele_presel_mask

    # Jets
    trig_jet_mask_clean = (abs(events.Jet.eta) < 2.4) & (events.Jet.jetId == 6)
    trig_jet_clean = events.Jet[trig_jet_mask_clean]

    results.steps["trig_jet"] = ak.sum(trig_jet_clean.pt > 25, axis=1) >= 3

    # HT
    trig_ht = ak.fill_none(ak.sum(trig_jet_clean.pt, axis=1), 40)
    events = set_ak_column(events, "trig_HT", trig_ht)

    # define btag mask
    btag_column = self.config_inst.x.btag_column
    b_score = trig_jet_clean[btag_column]
    # sometimes, jet b-score is nan, so fill it with 0
    #if ak.any(np.isnan(b_score)):
    #    b_score = ak.fill_none(ak.nan_to_none(b_score), 0)
    btag_mask = (b_score >= self.config_inst.x.btag_wp_score)

    # add btag steps
    events = set_ak_column(events, "cutflow.n_btag", ak.sum(btag_mask, axis=1))
    results.steps["nBjet1"] = events.cutflow.n_btag >= 1
    results.steps["nBjet2"] = events.cutflow.n_btag >= 2
    if self.config_inst.x("n_btag", 0) > 2:
        results.steps[f"nBjet{self.config_inst.x.n_btag}"] = events.cutflow.n_btag >= self.config_inst.x.n_btag

    results.steps["SR_mu"] = (results.steps["TrigMuPreselMask"]) & (results.steps["trig_jet"]) & (results.steps["nBjet1"])
    results.steps["SR_ele"] = (results.steps["TrigElePreselMask"]) & (results.steps["trig_jet"]) & (results.steps["nBjet1"])
    results.steps["SR"] = (results.steps["SR_mu"]) | (results.steps["SR_ele"])

    if (self.dataset_inst.is_data==False): #everything is "is_hbw" --> or self.dataset_inst.x("is_hbw", True):
        # Higgs mass
        gen_part_select = events.GenPart
        higgs_mask = (gen_part_select.pdgId == 25) & (gen_part_select.hasFlags("isHardProcess"))
        gen_h1 = gen_part_select[higgs_mask][:,0]
        gen_h2 = gen_part_select[higgs_mask][:,1]
        trig_mhh = (gen_h1 + gen_h2).mass
        events = set_ak_column(events, "trig_mHH", trig_mhh)

    #Horribly WONG TODO brute force adding masks to avoid errors
    results.event = mu_presel_mask
    events = set_ak_column(events, "Muon.is_tight", mu_cleaning_mask)
    events = set_ak_column(events, "Electron.is_tight", ele_cleaning_mask)
    #results.steps.SR = mu_presel_mask
    results.steps.Fake = ele_presel_mask
    results.objects.Electron = DotDict({"Electron": ele_cleaning_mask})
    results.objects.Muon = DotDict({"Muon": mu_cleaning_mask})
    results.aux = DotDict({"jet_mask": trig_jet_mask_clean})
    results.steps.all_but_bjet = (mu_presel_mask) & (ele_presel_mask)
    jet_indices = masked_sorted_indices(trig_jet_mask_clean, events.Jet.pt)
    results.x.n_central_jets = ak.num(jet_indices)
    

    events, results = self[post_selection](events, results, stats, **kwargs)

    return events, results

@sl_trigger.init
def sl_trigger_init(self: Selector) -> None:

    configure_selector(self)

    for trigger_columns in self.config_inst.x.trigger.values():
        for column in trigger_columns:
            self.uses.add(f"HLT.{column}")

    self.uses.add(f"Jet.{self.config_inst.x.btag_column}")

    if (self.dataset_inst.is_data==False):
        self.uses.add("GenPart.*")
        self.produces.add("trig_mHH")

    self.uses.add(event_weights_to_normalize)
    self.produces.add(event_weights_to_normalize)