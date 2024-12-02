from enum import Enum



# [ 0 1   0 1   0 0    0]
#  thumb index middle rest


class emgHeroActions(Enum):
    REST = 0b0000001
    MIDDLE_FLEX = 0b0000010
    MIDDLE_EXTEND = 0b0000100
    INDEX_FLEX = 0b0001000
    INDEX_EXTEND = 0b0010000
    THUMB_FLEX = 0b0100000
    THUMB_EXTEND = 0b1000000
    THUMB_INDEX_FLEX = 0b0101000    
    THUMB_INDEX_EXTEND = 0b1010000
    INDEX_MIDDLE_FLEX = 0b0001010
    INDEX_MIDDLE_EXTEND = 0b0010100
    THUMB_INDEX_MIDDLE_FLEX = 0b0101010
    THUMB_INDEX_MIDDLE_EXTEND = 0b1010100

action_mapping = {0b0000001:0,
                  0b0000010:1,
                  0b0000100:2,
                  0b0001000:3,
                  0b0010000:4,
                  0b0100000:5,
                  0b1000000:6,
                  0b0101000:7,
                  0b1010000:8,
                  0b0001010:9,
                  0b0010100:10,
                  0b0101010:11,
                  0b1010100:12}

print(0b1010100)