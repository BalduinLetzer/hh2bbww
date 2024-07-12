# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    produces={"trig_bits", "trig_bits_orth"},
)
def trigger_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column where each bin corresponds to a certain trigger
    """
    
    arr = ak.singletons(np.zeros(len(events)))
    arr_orth = ak.singletons(np.zeros(len(events)))

    id = 1
    
    for channel, trig_cols in self.config_inst.x.trigger.items():
        ref_trig = self.config_inst.x.ref_trigger[channel]
        for trig_col in trig_cols: 
            trig_passed = ak.singletons(ak.nan_to_none(ak.where(events.HLT[trig_col], id, np.float64(np.nan))))
            trig_passed_orth = ak.singletons(ak.nan_to_none(ak.where((events.HLT[ref_trig] & events.HLT[trig_col]), id, np.float64(np.nan))))
            arr = ak.concatenate([arr, trig_passed], axis=1)
            arr_orth = ak.concatenate([arr_orth, trig_passed_orth], axis=1)
            id += 1

    events = set_ak_column(events, "trig_bits", arr)
    events = set_ak_column(events, "trig_bits_orth", arr_orth)

    return events

@trigger_prod.init 
def trigger_prod_init(self: Producer) -> None:
    
    for trigger_columns in self.config_inst.x.trigger.values():
        for column in trigger_columns:
            self.uses.add(f"HLT.{column}")
