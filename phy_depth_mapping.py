import numpy as np


def get_depth_info(date, probe, channel_group):
    depth_info = {}
    if date == '20150918':
        supra = np.arange(1, 14)
        gran = np.arange(14, 21)
        infra = np.arange(21, 33)

    elif date == '20150919':
        supra = np.arange(1, 10)
        gran = np.arange(10, 15)
        infra = np.arange(15, 33)

    elif date == '20150917':
        supra = np.arange(1, 16)
        gran = np.arange(16, 21)
        infra = np.arange(21, 33)

    elif date == '20150916':
        supra = np.arange(1, 14)
        gran = np.arange(14, 20)
        infra = np.arange(20, 33)

    elif date == '20150915':
        supra = np.arange(1, 12)
        gran = np.arange(12, 19)
        infra = np.arange(19, 33)

    elif date == '20150902':
        supra = np.arange(1, 17)
        gran = np.arange(17, 22)
        infra = np.arange(22, 33)

    elif date == '20150826':
        if channel_group == 0:
            supra = np.arange(1, 3)
            gran = np.arange(3, 7)
            infra = np.arange(7, 17)
        elif channel_group == 1:
            supra = np.arange(1, 4)
            gran = np.arange(4, 9)
            infra = np.arange(9, 17)

    elif date == '20150901':
        if channel_group == 0:
            supra = np.arange(1, 12)
            gran = np.arange(12, 15)
            infra = np.arange(15, 17)
        elif channel_group == 1:
            supra = np.arange(1, 10)
            gran = np.arange(10, 14)
            infra = np.arange(14, 17)

    elif date == '20150825':
        if channel_group == 0:
            supra = np.arange(1, 9)
            gran = np.arange(9, 12)
            infra = np.arange(12, 17)
        elif channel_group == 1:
            supra = np.arange(1, 10)
            gran = np.arange(10, 14)
            infra = np.arange(14, 17)

    for chan in supra:
        depth_info[chan] = 'supragranular'
    for chan in gran:
        depth_info[chan] = 'granular'
    for chan in infra:
        depth_info[chan] = 'infragranular'

    if probe == 'a1x32':
        map = {
            1: 28,
            2: 18,
            3: 24,
            4: 22,
            5: 29,
            6: 19,
            7: 25,
            8: 23,
            9: 30,
            10: 16,
            11: 31,
            12: 7,
            13: 26,
            14: 20,
            15: 12,
            16: 5,
            17: 27,
            18: 17,
            19: 8,
            20: 3,
            21: 14,
            22: 21,
            23: 15,
            24: 1,
            25: 10,
            26: 2,
            27: 11,
            28: 4,
            29: 13,
            30: 0,
            31: 9,
            32: 6
        }
    elif probe == 'a2x16':
        if channel_group == 0:
            map = {
                1: 15,  # 5
                2: 14,  # 17
                3: 13,  # 20
                4: 12,  # 3
                5: 11,  # 7
                6: 10,  # 21
                7:  9,  # 16
                8:  8,  # 7
                9:  7,  # 23
                10: 6,  # 2
                11: 5,  # 19
                12: 4,  # 4
                13: 3,  # 22
                14: 2,  # 0
                15: 1,  # 18
                16: 0   # 6
            }
        elif channel_group == 1:
            map = {
                1: 15,  # 27
                2: 14,  # 12
                3: 13,  # 8
                4: 12,  # 26
                5: 11,  # 14
                6: 10,  # 31
                7:  9,  # 15
                8:  8,  # 30
                9:  7,  # 10
                10: 6,  # 25
                11: 5,  # 11
                12: 4,  # 29
                13: 3,  # 13
                14: 2,  # 24
                15: 1,  # 9
                16: 0   # 28
            }

    remap_depth = {}
    for chan in map:
        remap_depth[map[chan]] = depth_info[chan]

    return remap_depth