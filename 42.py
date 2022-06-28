import skimage as si
from functools import reduce
import math
from decimal import Decimal
import re
x = 42
collist = []
theme = re.compile(r'^0x')


def stradd(x, y):
    return x+y


def decimal(lst):
    return list(map(lambda s: round(Decimal(str(s))*Decimal('255')), lst))


pice = Decimal(str(math.pi))*Decimal('2')/Decimal(str(x))
for t in range(8):
    for i in range(x):
        rad = pice * Decimal(str(i))
        a = round(float(Decimal(str(math.cos(rad)))
                  * Decimal('14')*Decimal(str(t))), 2)
        b = round(float(Decimal(str(math.sin(rad)))
                  * Decimal('14')*Decimal(str(t))), 2)
        temp = []
        for j in range(3, 8):
            rgb = si.color.lab2rgb([14*j, float(a), float(b)])
            rgb = decimal(rgb)
            rgb_to_hex = map(lambda s: theme.sub('', format(s, '02x')), rgb)
            hexcode = "#" + reduce(stradd, list(rgb_to_hex))
            temp.append(hexcode)
        collist.append(temp)

for t in range(2):
    for i in range(x):
        rad = pice * Decimal(str(i))
        a = round(float(Decimal(str(math.cos(rad)))
                  * Decimal('14')*Decimal(str(t))), 2)
        b = round(float(Decimal(str(math.sin(rad)))
                  * Decimal('14')*Decimal(str(t))), 2)
        temp = []
        for j in range(3):
            rgb = si.color.lab2rgb([14*j, float(a), float(b)])
            rgb = decimal(rgb)
            rgb_to_hex = map(lambda s: theme.sub('', format(s, '02x')), rgb)
            hexcode = "#" + reduce(stradd, list(rgb_to_hex))
            temp.append(hexcode)
        collist.append(temp)


for i in collist:
    print(i)
