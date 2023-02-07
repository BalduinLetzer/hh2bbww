# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbw.ml.simple import SimpleDNN


processes = [
    "ggHH_kl_1_kt_1_sl_hbbhww",
    "tt",
    "st",
    # "w_lnu",
    # "dy_lep"
    # "qcd",
]

custom_procweights = {
    "ggHH_kl_1_kt_1_sl_hbbhww": 1 / 1000,
    "tt": 1 / 1000,
    "st": 1 / 1000,
}

dataset_names = {
    "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
    # TTbar
    "tt_sl_powheg",
    # "tt_dl_powheg",
    "tt_fh_powheg",
    # # SingleTop
    # "st_tchannel_t_powheg",
    # "st_tchannel_tbar_powheg",
    # "st_twchannel_t_powheg",
    # "st_twchannel_tbar_powheg",
    # "st_schannel_lep_amcatnlo",
    # # "st_schannel_had_amcatnlo",
    # # WJets
    # "w_lnu_ht70To100_madgraph",
    # "w_lnu_ht100To200_madgraph",
    # "w_lnu_ht200To400_madgraph",
    # "w_lnu_ht400To600_madgraph",
    # "w_lnu_ht600To800_madgraph",
    # "w_lnu_ht800To1200_madgraph",
    # "w_lnu_ht1200To2500_madgraph",
    # "w_lnu_ht2500_madgraph",
    # # DY
    # "dy_lep_m50_ht70to100_madgraph",
    # "dy_lep_m50_ht100to200_madgraph",
    # "dy_lep_m50_ht200to400_madgraph",
    # "dy_lep_m50_ht400to600_madgraph",
    # "dy_lep_m50_ht600to800_madgraph",
    # "dy_lep_m50_ht800to1200_madgraph",
    # "dy_lep_m50_ht1200to2500_madgraph",
    # "dy_lep_m50_ht2500_madgraph",
}

input_features = (
    "ht",
    # "m_bb",
    "deltaR_bb",
)

default_cls_dict = {
    "folds": 5,
    "layers": [512, 512, 512],
    "learningrate": 0.00050,
    "batchsize": 2048,  # 131xxx
    "epochs": 200,
    "eqweight": True,
    "dropout": False,  # 0.50
    "processes": processes,
    "custom_procweights": custom_procweights,
    "dataset_names": dataset_names,
    "input_features": input_features,
    "store_name": "inputs1",
}

# derived model, usable on command line
default_dnn = SimpleDNN.derive("default", cls_dict=default_cls_dict)


cls_dict = default_cls_dict
cls_dict["epochs"] = 6

test_dnn = SimpleDNN.derive("test", cls_dict=cls_dict)