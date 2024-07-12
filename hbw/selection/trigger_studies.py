# coding: utf-8

"""
Selector for triggerstudies
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import, DotDict
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column

from hbw.selection.common import masked_sorted_indices, configure_selector, pre_selection, post_selection
from hbw.util import four_vec
from hbw.selection.sl_remastered import sl_lepton_selection
from hbw.selection.jet import jet_selection


np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses=(four_vec({"Muon", "Electron", "Jet"})) | {
        pre_selection, post_selection,
        "TrigObj.*",
    },
    produces={
            pre_selection, post_selection, 
            "trig_mu_pt", "trig_ele_pt", "trig_HT",
              },
    b_tagger= "deepjet",
    btag_wp= "medium",
    exposed=True,
)
def trigger_studies(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    
    # get preselection and object definition
    events, results = self[pre_selection](events, stats, **kwargs)

    events, lepton_results = self[sl_lepton_selection](events, stats, **kwargs)
    results += lepton_results

    events, jet_results = self[jet_selection](events, lepton_results, stats, **kwargs)
    results += jet_results

    # save trigger results
    for channel, trigger_columns in self.config_inst.x.trigger.items():
        for trigger_column in trigger_columns:
            results.steps[f"Trigger_{channel}_{trigger_column}"] = events.HLT[trigger_column]
        #results.steps[f"Trigger_{trigger_column}"] = events.HLT[trigger_column]

    # save lepton pt
    muon = events.Muon[results.aux["mu_mask_tight"]]
    electron = events.Electron[results.aux["e_mask_tight"]]

    events = set_ak_column(events, "trig_mu_pt", ak.fill_none(ak.firsts(muon.pt), -1))
    events = set_ak_column(events, "trig_ele_pt", ak.fill_none(ak.firsts(electron.pt), -1))

    # check for match with HLT object
    hlt_muon_obj_mask = events.TrigObj.id == 13
    hlt_electron_obj_mask = events.TrigObj.id == 11

    hlt_muon_mask = hlt_muon_obj_mask & ((events.TrigObj.filterBits & (1<<3)) > 0) 
    hlt_electron_mask = hlt_electron_obj_mask & ((events.TrigObj.filterBits & (1<<1)) > 0)
    hlt_muon = events.TrigObj[hlt_muon_mask]
    hlt_electron = events.TrigObj[hlt_electron_mask]
    mu_combo = ak.cartesian({"offl": muon, "trigobj": hlt_muon}, nested=True)
    e_combo = ak.cartesian({"offl": electron, "trigobj": hlt_electron}, nested=True)

    off_mu, hlt_mu = ak.unzip(mu_combo)
    off_ele, hlt_ele = ak.unzip(e_combo)

    dR_mu = off_mu.delta_r(hlt_mu)
    dR_ele = off_ele.delta_r(hlt_ele)

    mindR_mu = ak.min(dR_mu, axis=-1)
    mindR_ele = ak.min(dR_ele, axis=-1)

    results.steps["TrigMuMatch"] = ak.fill_none(mindR_mu < 0.2, False)
    results.steps["TrigEleMatch"] = ak.fill_none(mindR_ele < 0.2, False)

    if self.dataset_inst.has_tag("is_hbw"):
        
        # check if lepton from W decay
        genparts = events.GenPart
        gen_Ids = genparts.pdgId

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

        # Higgs mass
        gen_part_select = events.GenPart
        higgs_mask = (gen_part_select.pdgId == 25) & (gen_part_select.hasFlags("isHardProcess"))
        gen_h1 = gen_part_select[higgs_mask][:,0]
        gen_h2 = gen_part_select[higgs_mask][:,1]
        trig_mhh = (gen_h1 + gen_h2).mass

        events = set_ak_column(events, "trig_mHH", trig_mhh)

    muon_mask = ak.sum(muon.pt > 15, axis=1) >= 1
    electron_mask = ak.sum(electron.pt > 15, axis=1) >= 1

    results.steps["TrigMuMask"] = muon_mask
    results.steps["TrigEleMask"] = electron_mask

    results.objects["Electron"]["Electron"] = masked_sorted_indices((results.aux["e_mask_tight"] & (events.Electron.pt>15)), events.Electron.pt)
    results.objects["Muon"]["Muon"] = masked_sorted_indices((results.aux["mu_mask_tight"] & (events.Muon.pt>15)), events.Muon.pt)
    
    # HT
    jet_mask = results.aux["jet_mask"]
    jets = events.Jet[jet_mask]
    
    trig_ht = ak.fill_none(ak.sum(jets.pt, axis=1), 40)
    
    events = set_ak_column(events, "trig_HT", trig_ht)

    results.steps["SR_mu"] = (
        results.steps.cleanup &
        results.steps.nJet3 &
        results.steps.nBjet1 &
        results.steps.TrigMuMask
    )
    results.steps["SR_ele"] = (
        results.steps.cleanup &
        results.steps.nJet3 &
        results.steps.nBjet1 &
        results.steps.TrigEleMask
    )
    results.steps["all_but_bjet"] = (
        results.steps.cleanup &
        results.steps.nJet3 &
        results.steps.nBjet1 &
        (results.steps.TrigMuMask | results.steps.TrigEleMask )
    )

    results.steps["all"] = results.event =  (
        results.steps.SR_ele | results.steps.SR_mu
    )

    events, results = self[post_selection](events, results, stats, **kwargs)

    return events, results

@trigger_studies.init
def trigger_init(self: Selector) -> None:

    if not getattr(self, "config_inst", None) or not getattr(self, "dataset_inst", None):
        return

    configure_selector(self)

    self.uses.add(sl_lepton_selection)
    self.uses.add(jet_selection)

    self.produces.add(sl_lepton_selection)
    self.produces.add(jet_selection)
    
    for trigger_columns in self.config_inst.x.trigger.values():
        for column in trigger_columns:
            self.uses.add(f"HLT.{column}")

    if self.dataset_inst.has_tag("is_hbw"):
        self.uses.add("GenPart.*")
        self.produces.add("trig_mHH")



