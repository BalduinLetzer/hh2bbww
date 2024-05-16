# coding: utf-8

"""
Event weight producer.
"""

import law

from columnflow.util import maybe_import
from columnflow.weight import WeightProducer, weight_producer
from columnflow.config_util import get_shifts_from_sources
from columnflow.columnar_util import Route

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@weight_producer(uses={"normalization_weight"}, mc_only=True)
def norm(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.normalization_weight


@weight_producer(mc_only=True)
def no_weights(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.Array(np.ones(len(events), dtype=np.float32))


@weight_producer(
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=True,
)
def base(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float32))
    for column in self.weight_columns.keys():
        weight = weight * Route(column).apply(events)

    return events, weight


@base.init
def base_init(self: WeightProducer) -> None:
    if not getattr(self, "config_inst", None) or not getattr(self, "dataset_inst", None):
        return

    year = self.config_inst.campaign.x.year

    if not self.weight_columns:
        raise Exception("weight_columns not set")

    if self.dataset_inst.has_tag("skip_scale"):
        # remove dependency towards mur/muf weights
        for column in [
            "normalized_mur_weight", "normalized_muf_weight", "normalized_murf_envelope_weight",
            "mur_weight", "muf_weight", "murf_envelope_weight",
        ]:
            self.weight_columns.pop(column, None)

    if self.dataset_inst.has_tag("skip_pdf"):
        # remove dependency towards pdf weights
        for column in ["pdf_weight", "normalized_pdf_weight"]:
            self.weight_columns.pop(column, None)

    if not self.dataset_inst.has_tag("is_ttbar"):
        # remove dependency towards top pt weights
        self.weight_columns.pop("top_pt_weight", None)

    self.shifts = set()
    for weight_column, shift_sources in self.weight_columns.items():
        shift_sources = law.util.make_list(shift_sources)
        shift_sources = [s.format(year=year) for s in shift_sources]
        shifts = get_shifts_from_sources(self.config_inst, *shift_sources)
        for shift in shifts:
            if weight_column not in shift.x("column_aliases").keys():
                # make sure that column aliases are implemented
                raise Exception(
                    f"Weight column {weight_column} implements shift {shift}, but does not use it"
                    f"in 'column_aliases' aux {shift.x('column_aliases')}",
                )

            # declare shifts that the produced event weight depends on
            self.shifts |= set(shifts)

    # store column names referring to weights to multiply
    self.uses |= self.weight_columns.keys()


btag_uncs = [
    "hf", "lf", "hfstats1_{year}", "hfstats2_{year}",
    "lfstats1_{year}", "lfstats2_{year}", "cferr1", "cferr2",
]

base.derive("default", cls_dict={"weight_columns": {
    "normalization_weight": [],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_weight": ["mu_sf"],
    "electron_weight": ["e_sf"],
    "normalized_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_murf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "top_pt_weight": ["top_pt"],
}})

base.derive("no_btag_weight", cls_dict={"weight_columns": {
    "normalization_weight": [],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_weight": ["mu_sf"],
    "electron_weight": ["e_sf"],
    "normalized_murf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "top_pt_weight": ["top_pt"],
}})

base.derive("btag_not_normalized", cls_dict={"weight_columns": {
    "normalization_weight": [],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_weight": ["mu_sf"],
    "electron_weight": ["e_sf"],
    "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_murf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "top_pt_weight": ["top_pt"],
}})

base.derive("btag_njet_normalized", cls_dict={"weight_columns": {
    "normalization_weight": [],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_weight": ["mu_sf"],
    "electron_weight": ["e_sf"],
    "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_murf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "top_pt_weight": ["top_pt"],
}})