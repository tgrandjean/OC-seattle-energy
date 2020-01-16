# -*- coding: utf-8 -*-


class Converter(object):

    @classmethod
    def kbtu_to_kwh(cls, energy):
        """Conversion from kilo british thermal unit to kilo watt hour.

        As a reminder : 3412 kwh = 1 btu
        more precisly : 3412.1416299 kwh = 1 btu
        So we multiply the energy value by thousand and then we divide the
        all by 3412 to obtain the value of energy in kwh.

        :args:
            energy (float, int) : the value of energy in kbtu
                                  (kilo british thermal unit)
        :returns:
            energy (float) : the value of energy converted in kwh
        :example:
            >>>ktbu_to_kwh(3.412)
            1.0
        """
        return (energy * 1e3) / 3412.1416299

    @classmethod
    def sf_to_square_m(cls, area):
        """Conversion from square foot to square meters.

        As a reminder : 10.764 sf = 1 m2
        Conversion rate : 10.76391041671
        So we divide the area value in square foot by 10.764 to obtain the
        area value in square meters.

        :args:
            area (float, in): the value of area in (sf) square foot
        :returns:
            area (float) : the value of area converted in m2
        """
        return area / 10.76391041671

    @classmethod
    def kbtu_per_sf_to_kwh_per_sm(cls, value):
        """Conversion from square foot to square meters.

        :args:
            area (float, in): energy density in KBtu/sf
        :returns:
            area (float) : the value of energy density converted in kwh/m2
        """
        return cls.kbtu_to_kwh(value) / cls.sf_to_square_m(1)
