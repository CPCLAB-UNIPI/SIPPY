from typing import Literal, TypedDict

IOMethods = Literal[
    "FIR",
    "ARX",
    "ARMA",
    "ARMAX",
    "OE",
    "ARARX",
    "ARARMAX",
    "BJ",
    "GEN",
    "EARMAX",
    "EOE",
]

OLSimMethods = Literal["CVA", "MOESP", "N4SID"]
PARSIMMethods = Literal["PARSIM_K", "PARSIM_P", "PARSIM_S"]
# TODO: Use | if issue resolved https://github.com/python/typing/issues/1932
SSMethods = Literal[
    "CVA", "MOESP", "N4SID", "PARSIM_K", "PARSIM_P", "PARSIM_S"
]
# TODO: Use | if issue resolved https://github.com/python/typing/issues/1932
AvailableMethods = Literal[
    "FIR",
    "ARX",
    "ARMA",
    "ARMAX",
    "OE",
    "ARARX",
    "ARARMAX",
    "BJ",
    "GEN",
    "EARMAX",
    "EOE",
    "CVA",
    "MOESP",
    "N4SID",
    "PARSIM_K",
    "PARSIM_P",
    "PARSIM_S",
]

ArxMethods = Literal["FIR", "ARX", "ARMA", "ARARX", "ARARMAX"]
OptMethods = Literal[
    "OE", "BJ", "GEN", "ARARX", "ARARMAX", "ARMA", "ARMAX", "EARMAX", "EOE"
]

RLSMethods = Literal["ARX", "ARMA", "ARMAX", "ARARX", "ARARMAX", "FIR", "OE"]

ICMethods = Literal["AIC", "AICc", "BIC"]

AvailableModes = Literal["LLS", "ILLS", "RLLS", "OPT"]

Flags = Literal["arx", "armax", "rls", "opt"]

CenteringMethods = Literal["InitVal", "MeanVal", None]


class OrderRanges(TypedDict, total=False):
    na_ord: tuple[int, int]
    nb_ord: tuple[int, int]
    nc_ord: tuple[int, int]
    nd_ord: tuple[int, int]
    nf_ord: tuple[int, int]
    delays: tuple[int, int]


class FlexOrderParams(TypedDict, total=False):
    IC: ICMethods
    id_mode: AvailableModes
    centering: CenteringMethods
