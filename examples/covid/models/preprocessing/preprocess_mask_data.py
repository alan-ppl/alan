import numpy as np
import pandas as pd
# import theano
import torch as t

from .preprocessed_data import PreprocessedData


def naive_set_region_days(df, out, col, rs, Ds):
    for i, r in enumerate(rs):
        for j, d in enumerate(Ds):
            out[i, j] = df.loc[r, d][col]
    return out


def naive_set_region_cms_days(df, out, cols, rs, Ds):
    for i, r in enumerate(rs):
        for k, d in enumerate(Ds):
            out[i, :, k] = df.loc[r, d][cols].values

    return out


# moving average
def smooth_variable(x, window_len=7):
    l = window_len
    s = np.r_[x[l - 1 : 0 : -1], x, x[-2 : -l - 1 : -1]]
    w = np.ones(window_len, "d")

    return np.convolve(w / w.sum(), s, mode="valid")


# Daily changes, smoothing, masking
def process_response(data, res, window, smooth):
    New = getattr(data, res)

    if res == "NewCases":
        # skip first day (since diff = nan there)
        New[:, 1:] = np.diff(data.Confirmed, axis=-1)
        start = 10
    elif res == "NewDeaths":
        New[:, 1:] = np.diff(data.Deaths, axis=-1)
        start = 20

    if smooth:
        for r in range(len(data.Rs)):
            New[r, :] = smooth_variable(New[r, :], window)[: len(data.Ds)]

    # mask first 10/20 days
    New[:, :start] = np.ma.masked
    # mask last 20 days
    New[:, -20:] = np.ma.masked

    # blank out negative cases
    New[New < 0] = np.nan
    # blank out nans and infs
    New = np.ma.masked_invalid(New)

    setattr(data, res, New)
    return data


def mask_us(NewCases, rs, days=32):
    states = [
        "Alaska",
        "Alabama",
        "Arkansas",
        "Arizona",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia-US",
        "Hawaii",
        "Iowa",
        "Idaho",
        "Illinois",
        "Indiana",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Massachusetts",
        "Maryland",
        "Maine",
        "Michigan",
        "Minnesota",
        "Missouri",
        "Mississippi",
        "Montana",
        "North Carolina",
        "North Dakota",
        "Nebraska",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "Nevada",
        "New York",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Virginia",
        "Vermont",
        "Washington",
        "Wisconsin",
        "West Virginia",
        "Wyoming",
    ]
    if set(states).issubset(set(rs)):
        inds = [rs.index(s) for s in states]

        for r_i in inds:
            NewCases[r_i, :days] = np.ma.masked

    return NewCases


def switch_mob(data):
    mob_feature = "avg_mobility_no_parks_no_residential"
    i = data.CMs.index(mob_feature)
    data.CMs[i] = data.CMs[-2]
    data.CMs[-2] = mob_feature

    mob_cms = data.ActiveCMs[:, i, :].copy()
    data.ActiveCMs[:, i, :] = data.ActiveCMs[:, -2, :]
    data.ActiveCMs[:, -2, :] = mob_cms

    return data


def init_data_object(df, data, Rs, CMs, Ds, window=7, smooth=False, mandateRun=False):
    CasesMask = False * np.ones((len(Rs), len(Ds)), dtype=bool)
    DeathsMask = False * np.ones((len(Rs), len(Ds)), dtype=bool)

    npi_names = CMs
    print(npi_names)
    nCMs = len(npi_names)

    # data.Active
    data.ActiveCMs = np.zeros((len(Rs), nCMs, len(Ds)))
    data.CMs = npi_names
    # data.RNames
    data.Rs = Rs
    data.Ds = Ds

    data.ActiveCMs = naive_set_region_cms_days(df, data.ActiveCMs, npi_names, Rs, Ds)

    NewCases = np.zeros((len(Rs), len(Ds)))
    data.NewCases = np.ma.array(NewCases)
    data.Confirmed = np.ma.array(np.zeros_like(NewCases))
    data.Confirmed = naive_set_region_days(df, data.Confirmed, "ConfirmedCases", Rs, Ds)
    data = process_response(data, "NewCases", window, smooth)

    if not mandateRun:
        print("Masking May cases in US states")
        data.NewCases = mask_us(data.NewCases, Rs)

    NewDeaths = np.zeros((len(Rs), len(Ds)))
    data.NewDeaths = np.ma.array(NewDeaths)
    data.Deaths = np.ma.array(np.zeros_like(NewDeaths))
    data.Deaths = naive_set_region_days(df, data.Deaths, "ConfirmedDeaths", Rs, Ds)
    data = process_response(data, "NewDeaths", window, smooth)
    data.NewDeaths = mask_us(data.NewDeaths, Rs)

    mob_feature = "avg_mobility_no_parks_no_residential"
    ismob = mob_feature in data.CMs
    wrong_mob_index = data.CMs[-2] != mob_feature
    if ismob and wrong_mob_index:
        data = switch_mob(data)

    return data


class Preprocess_masks(object):
    def __init__(
        self,
        path=f"../data/modelling_set/master_data_mob_us_True_m_w.csv",
        mandateRun=False,
    ):
        # retrieve data
        df = pd.read_csv(path)
        df = df.rename(columns={"date": "Date"})
        # del df['Unnamed: 0']
        self.Rs = df.country.unique().tolist()
        self.Ds = df.Date.unique().tolist()
        cols = self.oxcgrt_cols() + ["percent_mc"]
        if self.mob_cols()[0] in df.columns:
            cols += self.mob_cols()

        self.CMs = cols
        CMs_wearing = "percent_mc"
        del self.CMs[self.CMs.index("percent_mc")]
        self.CMs.append(CMs_wearing)
        col_names = self.CMs + ["country", "Date", "ConfirmedCases", "ConfirmedDeaths"]
        df = df[col_names]
        df = df.set_index(["country", "Date"])
        self.df = df

        # to be produced after featurization:
        self.data = None
        # to be defined at featurization:
        self.smooth = True
        self.mandateRun = mandateRun

    @property
    def nCMs(self):
        return len(self.CMs)

    @property
    def nRs(self):
        return len(self.Rs)

    @property
    def nDs(self):
        return len(self.Ds)

    @property
    def spikes_dict(self):
        dict = {
            "Costa Rica": [142, 143],
            "Ethiopia": [60, 61, 62, 63, 64, 65, 66, 67, 68],
            "Guatemala": [78],
            "Lebanon": [95, 96],
            "Libya": [114, 115],
            "Michigan": [112, 113, 120, 121],
            "United Kingdom": [61, 62],
            "Honduras": [19, 20, 21, 22, 23, 28, 29],
            "Netherlands": [102, 103],
            "Panama": [44, 45],
            "Singapore": [96],
            "Serbia": [85, 86],
            "Alabama": [57, 58],
            "Arizona": [59],
            "Colorado": [126, 127],
            "Delaware": [22, 23],
            "Minnesota": [64],
            "New Mexico": [22, 23],
            "Oregon": [36, 37, 38, 43, 44],
            "South Carolina": [33, 34, 35, 36, 37],
            "Washington": [22, 23],
            "Wisconsin": [109, 110],
            "Iowa": [118],
        }
        return dict

    def oxcgrt_cols(self):
        return [
            "C1_School closing",
            "C1_School closing_full",
            "C2_Workplace closing",
            "C2_Workplace closing_full",
            "C4_Restrictions on gatherings_3plus",
            "C6_Stay at home requirements",
            "C7_Restrictions on internal movement",
            "C4_Restrictions on gatherings_2plus",
            "C4_Restrictions on gatherings_full",
            "H6_Facial Coverings",
            "H6_Facial Coverings_3plus",
        ]

    def mob_cols(self):
        return [
            "avg_mobility_no_parks_no_residential",
            "residential_percent_change_from_baseline",
        ]

    def drop_npi_by_index(self, npi_index):
        self.df = self.df.drop(self.CMs[npi_index], 1)
        del self.CMs[npi_index]

    def drop_country_by_index(self, country_index):
        df_copy = self.df.reset_index()
        inds = list(df_copy[df_copy["country"] == self.Rs[country_index]].index)
        df_copy.drop(inds, inplace=True)
        df_copy = df_copy.set_index(["country", "Date"])
        self.df = df_copy
        del self.Rs[country_index]

    def drop_countries(self, mobility, country_leavout_inds=None):
        country_leavouts = [
            "Tanzania",
            "Benin",
            "Burkina Faso",
            "Cambodia",
            "Mali",
            "Nicaragua",
            "Oman",
            "Sudan",
            "Chile",
            "Cameroon",
            "Ecuador",
            "France",
            "Ghana",
            "Haiti",
            "Kazakhstan",
            "Pakistan",
            "Idaho",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Mississippi",
            "Tennessee",
            "Belgium",
            "Bulgaria",
            "Bosnia and Herzegovina",
            "Denmark",
            "Spain",
            "Peru",
            "Connecticut",
            "New Hampshire",
            "Rhode Island",
            "West Virginia",
            "Kuwait",
            "Senegal",
            "Mozambique",
            "Qatar"

        ]  # ["Jordan", "Bulgaria", "Australia"]

        if mobility != "exclude":
            missing_mob_days = ["Afghanistan", "Serbia"]
            country_leavouts += missing_mob_days

        if country_leavout_inds is None:
            country_leavout_inds = []

        for r in country_leavouts:
            if r in self.Rs:
                i = self.Rs.index(r)
                country_leavout_inds += [i]

        country_leavout_inds = sorted(list(set(country_leavout_inds)))
        for i in reversed(country_leavout_inds):
            # print(f"Excluding: {self.Rs[i]}")
            self.drop_country_by_index(i)

    def mask_all_spikes(self, case_object):
        assert self.Ds[0] == "2020-05-01"

        for region in self.spikes_dict.keys():
            if region in self.Rs:
                r_i = self.Rs.index(region)
                for i in self.spikes_dict[region]:
                    case_object[r_i, i] = np.ma.masked
        return case_object

    def threshold_countries_via_cases(self, min_cases):
        def find_cum_cases(r):
            return (
                self.df.loc[r]["ConfirmedCases"][-1]
                - self.df.loc[r]["ConfirmedCases"][0]
            )

        country_leavout_inds = []
        for r_i, r in enumerate(self.Rs):
            if find_cum_cases(r) <= min_cases:
                country_leavout_inds.append(r_i)

        for i in reversed(country_leavout_inds):
            # print(
            #     f"Excluding: {self.Rs[i]} because it has fewer than {min_cases} cumulative cases"
            # )
            self.drop_country_by_index(i)

    def featurize(
        self,
        masks="wearing",
        gatherings=3,
        mobility="include",
        smooth=False,
        drop_rs=True,
        n_mandates=2,
        country_leavout_inds=None,
        min_cases=5000,
        npi_leaveout_inds=None,
        mask_leave_on=False,
        reopening=True,
        produce_linear_increase_wearing_npi=False,
        start_date='2020-05-01',
        end_date='2020-09-21'
    ):
        if masks == "wearing":
            if "H6_Facial Coverings" in self.CMs:
                mandate_ind = self.CMs.index("H6_Facial Coverings")
                self.drop_npi_by_index(mandate_ind)
            if "H6_Facial Coverings_3plus" in self.CMs:
                mandate_ind = self.CMs.index("H6_Facial Coverings_3plus")
                self.drop_npi_by_index(mandate_ind)
        if masks == "mandate":
            wearing_ind = self.CMs.index("percent_mc")
            self.drop_npi_by_index(wearing_ind)

        if gatherings == 1:
            two_plus_ind = self.CMs.index("C4_Restrictions on gatherings_2plus")
            self.drop_npi_by_index(two_plus_ind)
            full_ind = self.CMs.index("C4_Restrictions on gatherings_full")
            self.drop_npi_by_index(full_ind)

        if mobility == "include":
            if "residential_percent_change_from_baseline" in self.CMs:
                mob_ind = self.CMs.index("residential_percent_change_from_baseline")
                self.drop_npi_by_index(mob_ind)

        if mobility == "exclude":
            if "avg_mobility_no_parks_no_residential" in self.CMs:
                mob_ind_1 = self.CMs.index("avg_mobility_no_parks_no_residential")
                self.drop_npi_by_index(mob_ind_1)
            if "residential_percent_change_from_baseline" in self.CMs:
                mob_ind_2 = self.CMs.index("residential_percent_change_from_baseline")
                self.drop_npi_by_index(mob_ind_2)

        if mobility == "only":
            assert "percent_mc" in self.CMs
            self.df = self.df[self.df.columns[-5:]]
            self.CMs = self.CMs[-3:]
            print(self.CMs, self.df.columns.tolist())

            assert len(self.CMs) == len(self.df.columns.tolist()) - 2

        # we'll smooth later, while making the data object
        self.smooth = smooth
        if drop_rs:
            self.drop_countries(mobility, country_leavout_inds)
            self.threshold_countries_via_cases(min_cases)

        if npi_leaveout_inds is not None:
            for i in reversed(npi_leaveout_inds):
                print(f"Omitting: {self.CMs[i]}")
                self.drop_npi_by_index(i)

        self.Ds = self.df.reset_index().Date.unique().tolist()
        # we'll do these later too
        self.mask_leave_on = mask_leave_on
        self.reopening = reopening
        self.produce_linear_increase_wearing_npi = produce_linear_increase_wearing_npi
        self.start_date = start_date
        self.end_date = end_date

    def keep_npi_on(self, npi_ind):
        nRs, nCMs, nDs = self.data.ActiveCMs.shape

        for r_i in range(nRs):
            row_we_want = self.data.ActiveCMs[r_i, npi_ind, :]
            activated_inds = [i for i in range(nDs) if row_we_want[i] == 1]
            if len(activated_inds) == 0:
                pass
            else:
                self.data.ActiveCMs[r_i, npi_ind, activated_inds[0]:] = 1.0

    def make_linear_copy(self, object):
        l = len(object)
        start = np.mean(object[:7])
        end = np.mean(object[-7:])
        new_object = np.linspace(start, end, l)
        return new_object

    def make_linear_increase_wearing_npi(self):
        npi_to_copy = 'percent_mc'
        npi_ind_to_copy = self.CMs.index(npi_to_copy)

        nRs, _, nDs = self.data.ActiveCMs.shape

        for r_i in range(nRs):
            self.data.ActiveCMs[r_i, npi_ind_to_copy, :] = self.make_linear_copy(self.data.ActiveCMs[r_i, npi_ind_to_copy, :])

    def center_npis(self):
        # which CMs do we want to zero-out?
        npi_set_to_zero_out = ['C1_School closing',
                               'C1_School closing_full',
                               'C2_Workplace closing',
                               'C2_Workplace closing_full',
                               'C4_Restrictions on gatherings_3plus',
                               'C6_Stay at home requirements',
                               'C7_Restrictions on internal movement',
                               'C4_Restrictions on gatherings_2plus',
                               'C4_Restrictions on gatherings_full']
                               # 'H6_Facial Coverings',
                               # 'H6_Facial Coverings_3plus']

        npis = [CM for CM in self.CMs if CM in npi_set_to_zero_out]
        npi_inds = [self.CMs.index(npi) for npi in npis]

        nRs, _, _ = self.data.ActiveCMs.shape

        for r_i in range(nRs):
            for npi in npi_inds:
                if self.data.ActiveCMs[r_i, npi, 0] == 1:
                    self.data.ActiveCMs[r_i, npi, :] = self.data.ActiveCMs[r_i, npi, :] - 1

    def change_dates(self, sd, ed):
        if sd == '2020-05-01' and ed == '2020-09-21':
            return

        sd_ind = self.Ds.index(sd)
        ed_ind = self.Ds.index(ed)

        self.data.ActiveCMs = self.data.ActiveCMs[:, :, sd_ind:ed_ind]
        print(self.data.NewCases.mask)
        self.data.NewCases = self.data.NewCases[:, sd_ind:ed_ind]
        # mask first 10/20 days
        self.data.NewCases[:, :10] = np.ma.masked
        # mask last 20 days
        self.data.NewCases[:, -20:] = np.ma.masked
        print(self.data.NewCases.mask)
        self.Ds = self.Ds[sd_ind:ed_ind]
        self.data.Ds = self.data.Ds[sd_ind:ed_ind]


    def make_preprocessed_object(self):
        # initialise data object with old data, to be filled with new data via init_data_object
        #BRAUNER_PATH = "../data/modelling_set/data_final_nov.csv"
        #data_init = preprocess_data(BRAUNER_PATH, last_day="2020-05-30", smoothing=1)
        #data_init.mask_reopenings(print_out=False)
        data_init = PreprocessedData(
            [],#Active,
            [],#Confirmed,
            [],#ActiveCMs,
            [],#CMs,
            [],#sorted_regions,
            [],#Ds,
            [],#Deaths,
            [],#NewDeaths,
            [],#NewCases,
            []#region_full_names,
        )
        self.data = init_data_object(
            self.df,
            data_init,
            Rs=self.Rs,
            CMs=self.CMs,
            Ds=self.Ds,
            smooth=self.smooth,
            mandateRun=self.mandateRun,
        )
        self.data.NewCases = self.mask_all_spikes(self.data.NewCases)

        if self.mask_leave_on:
            if 'H6_Facial Coverings' in self.CMs:
                print('leaving masks on')
                npi_ind = self.CMs.index('H6_Facial Coverings')
                self.keep_npi_on(npi_ind)
            if 'H6_Facial Coverings_3plus' in self.CMs:
                npi_ind = self.CMs.index('H6_Facial Coverings_3plus')
                self.keep_npi_on(npi_ind)

        self.change_dates(self.start_date, self.end_date)

        if self.reopening:
            print('centering NPIs')
            self.center_npis()

        if self.produce_linear_increase_wearing_npi:
            self.make_linear_increase_wearing_npi()
