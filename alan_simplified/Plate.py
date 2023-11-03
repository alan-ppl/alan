class PlateTimeseries():
    def __init__(self, **kwargs):
        self.prog = kwargs

class Plate(PlateTimeseries):
    pass

class Timeseries(PlateTimeseries):
    pass

class Group():
    def __init__(self, **kwargs):
        #Groups can only contain variables, not Plates/Timeseries/other Groups.
        for dist in kwargs.values():
            assert isinstance(dist, AlanDist)

        self.prog = kwargs
