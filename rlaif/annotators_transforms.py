from typing import Dict, List

import numpy as np


class MessageTransform:
    def __call__(self, pair: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Pass through all the keys, if any
        dict_to_return = {k: v for k, v in pair.items() if k != "message"}
        dict_to_return["message"] = np.array([["".join([chr(c) for c in row if c != 0]) for row in message]
                                             for message in pair['message']])
        return dict_to_return


class BlstatsTransform:
    def __init__(self, blstats_keys: List[str]):
        self.blstats_keys = blstats_keys
        self.hunger_num_to_str = {
          0: "Satiated", 1: "", 2: "Hungry", 3: "Weak",
          4: "Fainting", 5: "Fainted ", 6: "Starved"
        }
        self.blstats_to_index = {
            "NLE_BL_X": (0, "X:{}"),
            "NLE_BL_Y": (1, "Y:{}"),
            "NLE_BL_STR25": (2, "Str:{}"),
            "NLE_BL_STR125": (3, "Str:{}"),
            "NLE_BL_DEX": (4, "Dex:{}"),
            "NLE_BL_CON": (5, "Con:{}"),
            "NLE_BL_INT": (6, "Int:{}"),
            "NLE_BL_WIS": (7, "Wis:{}"),
            "NLE_BL_CHA": (8, "Cha:{}"),
            "NLE_BL_SCORE": (9, "Score:{}"),
            "NLE_BL_HP": (10, "HP:{}"),
            "NLE_BL_HPMAX": (11, "({})"),
            "NLE_BL_DEPTH": (12, "Dlvl:{}"),
            "NLE_BL_GOLD": (13, "$:{}"),
            "NLE_BL_ENE": (14, "Ene:{}"),
            "NLE_BL_ENEMAX": (15, "Em:{}"),
            "NLE_BL_AC": (16, "AC:{}"),
            "NLE_BL_HD": (17, "HD:{}"),
            "NLE_BL_XP": (18, "Xp:{}"),
            "NLE_BL_EXP": (19, "/{}"),
            "NLE_BL_TIME": (20, "T:{}"),
            "NLE_BL_HUNGER": (21, "{}"),
            "NLE_BL_CAP": (22, "Cap:{}"),
            "NLE_BL_DNUM": (23, "Dn:{}"),
            "NLE_BL_DLEVEL": (24, "Lvl:{}"),
            "NLE_BL_CONDITION": (25, "Cnd:{}"),
            "NLE_BL_ALIGN": (26, "Algn:{}"),
        }

    def blstats_to_str(self, blstats: np.ndarray):
        """Process an individual blstat"""
        bls = " ".join([self.blstats_to_index[key][1].format(blstats[self.blstats_to_index[key][0]]
                                                             if key != "NLE_BL_HUNGER"
                                                             else self.hunger_num_to_str[int(blstats[self.blstats_to_index[key][0]])])
                        for key in self.blstats_keys])
        return bls

    def __call__(self, pair: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Pass through all the keys, if any
        dict_to_return = {k: v for k, v in pair.items() if k != 'blstats'}
        # pair['blstats'] is (2, seq_len, bldim)
        dict_to_return['blstats'] = [[self.blstats_to_str(bls) for bls in seq] for seq in pair['blstats']]
        return dict_to_return
