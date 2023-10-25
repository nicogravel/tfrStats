Search.setIndex({"docnames": ["index", "modules", "notebooks/statistical_approach", "tfrStats", "tfrStats.cluster_correction", "tfrStats.get_dpvals_minmax", "tfrStats.get_dpvals_whole", "tfrStats.get_pvals_minmax", "tfrStats.get_pvals_whole", "tfrStats.load_uv_tfrs", "tfrStats.plot_dtfr_stats", "tfrStats.plot_tfr_stats", "tfrStats.tfr_spw_stats_minmax", "tfrStats.tfr_spw_stats_whole", "uvtfrs"], "filenames": ["index.rst", "modules.rst", "notebooks/statistical_approach.ipynb", "tfrStats.rst", "tfrStats.cluster_correction.rst", "tfrStats.get_dpvals_minmax.rst", "tfrStats.get_dpvals_whole.rst", "tfrStats.get_pvals_minmax.rst", "tfrStats.get_pvals_whole.rst", "tfrStats.load_uv_tfrs.rst", "tfrStats.plot_dtfr_stats.rst", "tfrStats.plot_tfr_stats.rst", "tfrStats.tfr_spw_stats_minmax.rst", "tfrStats.tfr_spw_stats_whole.rst", "uvtfrs.md"], "titles": ["<strong>Time Frequency Representation Statistics</strong>", "tfrStats", "<span class=\"section-number\">2. </span>A mini-tutorial", "tfrStats package", "tfrStats.cluster_correction module", "tfrStats.get_dpvals_minmax module", "tfrStats.get_dpvals_whole module", "tfrStats.get_pvals_minmax module", "tfrStats.get_pvals_whole module", "tfrStats.load_uv_tfrs module", "tfrStats.plot_dtfr_stats module", "tfrStats.plot_tfr_stats module", "tfrStats.tfr_spw_stats_minmax module", "tfrStats.tfr_spw_stats_whole module", "<span class=\"section-number\">1. </span>On TFR statistical assessments"], "terms": {"mini": 0, "tutori": 0, "assess": [0, 2], "signific": [0, 2], "tfr": [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13], "thi": [0, 2, 4, 5, 6, 7, 8, 9, 10, 11], "codebook": [0, 2], "simpl": [0, 2], "exampl": [0, 2, 12, 13], "meant": [0, 2], "illustr": [0, 2], "two": [0, 2, 9, 10, 11, 12, 13, 14], "fundament": [0, 2], "question": [0, 2], "analysi": [0, 2], "1": [0, 2, 9, 10, 11, 12, 13, 14], "permut": [0, 2, 7, 12, 13], "base": [0, 2, 4, 12, 13, 14], "null": [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "hypothesi": [0, 2], "test": [0, 2, 4, 13, 14], "2": [0, 2, 5, 7, 9, 10, 11, 12, 13, 14], "correct": [0, 4, 10, 11, 12, 13], "multipl": [0, 2, 10, 11], "comparison": [0, 2, 10, 11], "typic": [0, 2], "more": [0, 2], "than": [0, 2], "one": [0, 2], "approach": [0, 2, 12], "given": [0, 2], "scenario": [0, 2], "thei": [0, 2, 10, 11], "all": [0, 2, 4], "depend": [0, 2, 14], "dimens": [0, 2, 13], "data": [0, 2, 14], "hand": [0, 2], "spatial": [0, 2, 14], "locat": [0, 2, 14], "paramet": [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "trial": [0, 2], "differ": [0, 2, 4, 8], "condit": [0, 2, 9, 10, 11, 12, 13], "natur": [0, 2, 14], "etc": [0, 2], "For": [0, 2], "basic": [0, 2], "pedagog": [0, 2], "purpos": [0, 2], "here": [0, 2, 9, 10, 11, 12, 13], "i": [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "focu": [0, 2], "spectral": [0, 9, 10, 11, 12, 13], "power": [0, 9, 10, 11, 12, 13, 14], "increas": [0, 2], "rel": [0, 2], "baselin": [0, 2], "us": [0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13], "variant": [0, 2], "same": [0, 2], "method": [0, 2, 4, 5, 6, 7, 8, 14], "tradit": [0, 2], "min": [0, 4, 6, 7, 8, 9, 10, 11, 12, 13], "max": [0, 4, 6, 7, 8, 9, 10, 11, 12, 13], "distribut": [0, 2, 4, 5, 6, 7, 8, 12, 13], "which": [0, 2], "captur": [0, 2, 12], "variat": [0, 2, 12], "extrem": [0, 2, 12], "ditribut": [0, 2, 12], "hypthesi": [0, 2], "whole": [0, 4, 5, 8, 9, 10, 11, 12, 13], "obtain": [0, 2, 4], "averag": [0, 2, 4, 5, 6, 8, 13], "across": [0, 2], "specif": [0, 2], "invit": [0, 2], "reader": [0, 2], "audit": [0, 2], "code": [0, 2], "propvid": [0, 2], "feedback": [0, 2, 14], "comment": [0, 2], "open": [0, 2], "discuss": [0, 2], "subsect": [0, 2], "within": [0, 2, 4], "background": [0, 2], "section": [0, 2], "sinc": [0, 2], "ar": [0, 2, 4, 10, 11, 12, 13, 14], "sever": [0, 2], "wai": [0, 2], "achiev": [0, 2], "goal": [0, 2], "mani": [0, 2], "realiz": [0, 2, 8], "other": [0, 2], "relat": [0, 2, 14], "e": [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "threshold": [0, 2, 4, 10, 11], "mai": [0, 2], "obtaind": [0, 2], "from": [0, 2, 4, 5, 6, 7, 8, 12, 13], "percentil": [0, 2, 9, 10, 11, 12, 13], "directli": [0, 2], "further": [0, 2], "equival": [0, 2], "pool": [0, 2, 4, 5, 6, 7, 8, 12, 13], "accomplish": [0, 2], "among": [0, 2, 14], "common": [0, 2], "veri": [0, 2], "hope": [0, 2], "help": [0, 2], "those": [0, 2], "research": [0, 2], "includ": [0, 2], "myself": [0, 2], "need": [0, 2], "clariti": [0, 2], "matter": [0, 2], "touch": [0, 2], "ground": [0, 2], "bewar": [0, 2], "even": [0, 2], "error": [0, 2], "spot": [0, 2], "loop": [0, 2], "optim": [0, 2], "comput": [0, 4, 5, 6, 7, 8, 12, 13, 14], "p": [0, 4, 5, 6, 7, 8, 10, 11, 12, 13], "valu": [0, 4, 5, 6, 7, 8, 10, 11, 12, 13], "chang": [0, 2], "slightli": [0, 2], "In": [0, 2, 12], "minimum": [0, 2, 12], "maximum": [0, 2, 12], "each": [0, 2, 4, 5, 6, 7, 8, 12], "when": [0, 2, 13], "empir": [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13], "so": [0, 2], "preserv": [0, 2, 13], "onc": [0, 2], "have": [0, 2, 14], "been": [0, 2, 14], "cumul": [0, 2, 5, 6, 7, 8], "addition": [0, 2], "an": [0, 2], "option": [0, 2, 8, 10, 11], "step": [0, 2], "cluster": [0, 4, 10, 11, 12, 13, 14], "implement": [0, 2, 9, 10, 11, 12, 13], "provid": [0, 2, 14], "right": [0, 2], "environ": [0, 2], "instal": [0, 2], "jupyt": [0, 2], "notebook": [0, 2], "should": [0, 2], "work": [0, 2, 4, 8], "support": [0, 2], "clear": [0, 2], "The": [0, 2], "function": [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "exploratori": [0, 2], "state": [0, 2], "subject": [0, 2], "A": [0, 14], "pleas": [0, 2], "feel": [0, 2], "free": [0, 2], "improv": [0, 2], "can": [0, 2], "download": [0, 2], "http": [0, 2, 14], "github": [0, 2], "com": [0, 2], "nicogravel": [0, 2], "tfrstat": [0, 2], "On": 0, "refer": 0, "install": 0, "import": 0, "librari": 0, "defin": [0, 4], "bin": [0, 4, 5, 6, 7, 8, 10, 11], "depth": [0, 10], "space": [0, 4, 5, 6, 7, 8], "packag": 1, "submodul": 1, "cluster_correct": [1, 3, 10, 11], "modul": [1, 14], "get_dpvals_minmax": [1, 3], "get_dpvals_whol": [1, 3], "get_pvals_minmax": [1, 3, 10, 11], "get_pvals_whol": [1, 3, 5, 6, 7, 10, 11], "load_uv_tfr": [1, 3, 10, 11], "plot_dtfr_stat": [1, 2, 3], "plot_tfr_stat": [1, 2, 3], "tfr_spw_stats_minmax": [1, 3, 9], "tfr_spw_stats_whol": [1, 3], "content": 1, "python": 2, "statist": [2, 4, 5, 6, 7, 8, 12, 13], "To": 2, "run": 2, "clone": 2, "local": [2, 14], "fork": 2, "first": 2, "git": 2, "cd": 2, "pip": 2, "voil\u00e0": 2, "in_path": 2, "volum": 2, "gandalf": 2, "mwlamprop": 2, "out_path": 2, "result": [2, 4, 9, 10, 11], "stat": [2, 4, 10, 11], "univari": [2, 14], "we": [2, 12, 13], "plot": [2, 10, 11], "val": [2, 4], "0": [2, 4, 9, 10, 11, 12, 13], "05": [2, 4, 14], "top": 2, "panel": 2, "blue": 2, "trace": 2, "alpha": [2, 4, 10, 11, 12, 13, 14], "mask": 2, "specifi": 2, "contain": [2, 12, 13, 14], "abov": [2, 10, 11], "cutoff": 2, "95": 2, "bottom": 2, "red": 2, "01": 2, "cluster_s": [2, 4, 10, 11, 12, 13], "fband": [2, 6, 8, 9, 10, 11, 12, 13], "cond": [2, 9, 10, 11, 12, 13], "type": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "minmax": 2, "uvtfr_stats_high_grat_spw_1000_minmax": 2, "npz": [2, 9, 10, 11, 12, 13], "emp": 2, "30": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13], "12": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "16": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "113": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13], "1000": [2, 5, 7, 9, 10, 11, 12, 13], "75": 2, "96054268486813": 2, "3": [2, 9, 10, 11, 12, 13, 14], "uvtfr_stats_high_grat_spw_100": 2, "70": 2, "79162686024465": 2, "One": 2, "appli": 2, "either": 2, "percetnil": 2, "onli": 2, "singl": [2, 14], "pvalu": [2, 4], "turn": 2, "second": 2, "distanc": [2, 4, 10, 11], "between": [2, 4], "neighbour": [2, 4, 10, 11], "size": [2, 4, 10, 11, 12, 13, 14], "its": [2, 4], "below": [2, 4], "true": [2, 4], "assign": [2, 4], "acorrect": 2, "4": [2, 14], "5": [2, 14], "must": 2, "consid": 2, "over": 2, "predefin": 2, "well": [2, 10, 11], "diment": 2, "roi": 2, "therefor": [2, 14], "6": [2, 14], "peak": 2, "rang": 2, "28": [2, 14], "44": 2, "86": 2, "6390137791448": 2, "91082666125475": 2, "7": [2, 14], "52901730660238": 2, "8": [2, 14], "9": [2, 14], "sourc": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "nearest": 4, "l1": 4, "unweight": 4, "time": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "frequenc": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "center": 4, "If": 4, "adapt": [4, 8], "3d": 4, "arrai": [4, 8], "2d": 4, "map": [4, 5, 6, 7, 8], "add": [4, 8], "possibl": 4, "metric": 4, "float": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "un": 4, "g": [4, 14], "return": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "author": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "nicola": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "gravel": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "19": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "09": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "2023": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "tfr_emp": [5, 6, 7, 8], "tfr_null": [5, 6, 7, 8], "tail": [5, 7], "get": [5, 6, 7, 8], "merg": [5, 6, 7], "represent": [5, 6, 7, 8, 9, 10, 11, 12, 13], "null_tfr": [5, 6, 7, 8], "nul": [6, 7, 8], "n": 8, "dimension": 8, "input_path": [9, 10, 11, 12, 13], "svar": [9, 10, 11, 12, 13], "ob": [9, 10, 11, 12, 13], "load": 9, "handl": [9, 10, 11], "dictionari": [9, 10, 11], "string": [9, 10, 11, 12, 13], "path": [9, 10, 11, 12, 13], "file": [9, 10, 11, 12, 13], "index": [9, 10, 11, 12, 13], "int": [9, 10, 11, 12, 13], "gpr": [9, 10, 11, 12, 13], "band": [9, 10, 11, 12, 13, 14], "low": [9, 10, 11, 12, 13], "high": [9, 10, 11, 12, 13, 14], "higher": [9, 10, 11, 12, 13], "nulltyp": [9, 10, 11, 12, 13], "integer": [9, 10, 11, 12, 13], "100": [9, 10, 11, 12, 13, 14], "n_cond": [9, 10, 11, 12, 13], "x": [9, 10, 11, 12, 13], "n_site": [9, 10, 11, 12, 13], "n_freq": [9, 10, 11, 12, 13], "n_time": [9, 10, 11, 12, 13], "also": [10, 11], "alreadi": [10, 11], "taken": [10, 11], "account": [10, 11, 12, 13], "alltoghet": [10, 11], "n_perm": [12, 13], "asess": [12, 13], "truncat": 12, "keep": [12, 13], "record": 12, "site": 12, "It": [12, 14], "onset": [12, 13], "shift": [12, 13, 14], "triakl": [12, 13], "current": [12, 13, 14], "400": [12, 13], "ms": [12, 13], "window": [12, 13], "compatibilityu": [12, 13], "syncopi": [12, 13], "now": [12, 13], "reli": [12, 13], "ftpool_": [12, 13], "mat": [12, 13], "fieldtrip": [12, 13], "s": 14, "detail": 14, "ha": 14, "propos": 14, "electrophysiolog": 14, "brain": 14, "activ": 14, "most": 14, "stimulu": 14, "inform": 14, "visual": 14, "cortex": 14, "10": 14, "11": 14, "13": 14, "14": 14, "number": 14, "suggest": 14, "15": 14, "georgio": 14, "michalarea": 14, "julien": 14, "vezoli": 14, "stan": 14, "van": 14, "pelt": 14, "jan": 14, "mathij": 14, "schoffelen": 14, "henri": 14, "kennedi": 14, "pascal": 14, "fri": 14, "beta": 14, "gamma": 14, "rhythm": 14, "subserv": 14, "feedforward": 14, "influenc": 14, "human": 14, "cortic": 14, "area": 14, "neuron": 14, "89": 14, "384": 14, "397": 14, "2016": 14, "doi": 14, "1016": 14, "j": 14, "2015": 14, "018": 14, "christoph": 14, "m": 14, "lewi": 14, "conrado": 14, "bosman": 14, "brunet": 14, "bruss": 14, "lima": 14, "mark": 14, "robert": 14, "thilo": 14, "womelsdorf": 14, "peter": 14, "de": 14, "weerd": 14, "sergio": 14, "neuenschwand": 14, "wolf": 14, "singer": 14, "biorxiv": 14, "1101": 14, "049718": 14, "supratim": 14, "rai": 14, "nathan": 14, "crone": 14, "ernst": 14, "niebur": 14, "piotr": 14, "franaszczuk": 14, "steven": 14, "hsiao": 14, "neural": 14, "correl": 14, "oscil": 14, "60": 14, "200": 14, "hz": 14, "macaqu": 14, "field": 14, "potenti": 14, "implic": 14, "electrocorticographi": 14, "journal": 14, "neurosci": 14, "45": 14, "11526": 14, "11536": 14, "2008": 14, "hiroya": 14, "ono": 14, "masaki": 14, "sonoda": 14, "brian": 14, "h": 14, "silverstein": 14, "kaori": 14, "takafumi": 14, "kubota": 14, "aime": 14, "f": 14, "luat": 14, "rothermel": 14, "sandeep": 14, "sood": 14, "eishi": 14, "asano": 14, "spontan": 14, "clinic": 14, "neurophysiolog": 14, "132": 14, "2391": 14, "2403": 14, "2021": 14, "xiaoxuan": 14, "jia": 14, "matthew": 14, "smith": 14, "adam": 14, "kohn": 14, "select": 14, "coher": 14, "compon": 14, "31": 14, "25": 14, "9390": 14, "9403": 14, "2011": 14, "jeremi": 14, "r": 14, "man": 14, "joshua": 14, "jacob": 14, "itzhak": 14, "michael": 14, "kahana": 14, "broadband": 14, "spectra": 14, "spike": 14, "29": 14, "43": 14, "13613": 14, "13620": 14, "2009": 14, "1523": 14, "jneurosci": 14, "2041": 14, "d": 14, "herm": 14, "k": 14, "miller": 14, "b": 14, "wandel": 14, "winaw": 14, "cerebr": 14, "2951": 14, "2959": 14, "2014": 14, "1093": 14, "cercor": 14, "bhu091": 14, "reflect": 14, "imag": 14, "structur": 14, "neuroimag": 14, "635": 14, "643": 14, "2019": 14, "04": 14, "440025": 14, "timo": 14, "kerkoerl": 14, "w": 14, "self": 14, "bruno": 14, "dagnino": 14, "mari": 14, "alic": 14, "gariel": 14, "mathi": 14, "jasper": 14, "poort": 14, "chri": 14, "der": 14, "togt": 14, "pieter": 14, "roelfsema": 14, "character": 14, "process": 14, "monkei": 14, "proceed": 14, "nation": 14, "academi": 14, "scienc": 14, "111": 14, "40": 14, "14332": 14, "14341": 14, "1073": 14, "pna": 14, "1402773111": 14, "maryam": 14, "bijanzadeh": 14, "lauri": 14, "nurminen": 14, "sam": 14, "merlin": 14, "andrew": 14, "clark": 14, "alessandra": 14, "angelucci": 14, "distinct": 14, "laminar": 14, "global": 14, "context": 14, "primat": 14, "primari": 14, "259": 14, "274": 14, "e4": 14, "2018": 14, "08": 14, "020": 14, "lar": 14, "muckli": 14, "federico": 14, "martino": 14, "luca": 14, "vizioli": 14, "luci": 14, "petro": 14, "fraser": 14, "kamil": 14, "ugurbil": 14, "rainer": 14, "goebel": 14, "essa": 14, "yacoub": 14, "contextu": 14, "superfici": 14, "layer": 14, "v1": 14, "biologi": 14, "20": 14, "2690": 14, "2695": 14, "cub": 14, "057": 14, "andr\u00e9": 14, "mora": 14, "basto": 14, "arturo": 14, "oostenveld": 14, "jarrod": 14, "dowdal": 14, "exert": 14, "through": 14, "channel": 14, "85": 14, "390": 14, "401": 14, "benjamin": 14, "fischer": 14, "detlef": 14, "wegen": 14, "epidur": 14, "about": 14, "shape": 14, "color": 14, "commun": 14, "690": 14, "1038": 14, "s42003": 14, "021": 14, "02207": 14, "eric": 14, "nonparametr": 14, "eeg": 14, "meg": 14, "164": 14, "177": 14, "190": 14, "2007": 14, "jneumeth": 14, "03": 14, "024": 14, "c": 14, "pernet": 14, "latinu": 14, "t": 14, "nichol": 14, "rousselet": 14, "mass": 14, "analys": 14, "event": 14, "simul": 14, "studi": 14, "250": 14, "93": 14, "cut": 14, "edg": 14, "org": 14, "003": 14}, "objects": {"": [[3, 0, 0, "-", "tfrStats"]], "tfrStats": [[4, 0, 0, "-", "cluster_correction"], [5, 0, 0, "-", "get_dpvals_minmax"], [6, 0, 0, "-", "get_dpvals_whole"], [7, 0, 0, "-", "get_pvals_minmax"], [8, 0, 0, "-", "get_pvals_whole"], [9, 0, 0, "-", "load_uv_tfrs"], [10, 0, 0, "-", "plot_dtfr_stats"], [11, 0, 0, "-", "plot_tfr_stats"], [12, 0, 0, "-", "tfr_spw_stats_minmax"], [13, 0, 0, "-", "tfr_spw_stats_whole"]], "tfrStats.cluster_correction": [[4, 1, 1, "", "cluster_correction"]], "tfrStats.get_dpvals_minmax": [[5, 1, 1, "", "get_dpvals_minmax"]], "tfrStats.get_dpvals_whole": [[6, 1, 1, "", "get_dpvals_whole"]], "tfrStats.get_pvals_minmax": [[7, 1, 1, "", "get_pvals_minmax"]], "tfrStats.get_pvals_whole": [[8, 1, 1, "", "get_pvals_whole"]], "tfrStats.load_uv_tfrs": [[9, 1, 1, "", "load_uv_tfrs"]], "tfrStats.plot_dtfr_stats": [[10, 1, 1, "", "plot_dtfr_stats"]], "tfrStats.plot_tfr_stats": [[11, 1, 1, "", "plot_tfr_stats"]], "tfrStats.tfr_spw_stats_minmax": [[12, 1, 1, "", "tfr_spw_stats_minmax"]], "tfrStats.tfr_spw_stats_whole": [[13, 1, 1, "", "tfr_spw_stats_whole"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"time": [0, 2], "frequenc": [0, 2], "represent": 0, "statist": [0, 14], "content": [0, 3], "python": 0, "packag": [0, 2, 3], "tfrstat": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "A": 2, "mini": 2, "tutori": 2, "install": 2, "import": 2, "librari": 2, "defin": 2, "function": 2, "comput": 2, "p": 2, "valu": 2, "min": 2, "max": 2, "whole": 2, "null": 2, "cluster": 2, "correct": 2, "bin": 2, "spectral": 2, "power": 2, "depth": 2, "space": 2, "submodul": 3, "modul": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "cluster_correct": 4, "todo": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "get_dpvals_minmax": 5, "get_dpvals_whol": 6, "get_pvals_minmax": 7, "get_pvals_whol": 8, "load_uv_tfr": 9, "plot_dtfr_stat": 10, "plot_tfr_stat": 11, "tfr_spw_stats_minmax": 12, "tfr_spw_stats_whol": 13, "On": 14, "tfr": 14, "assess": 14, "background": 14, "open": 14, "discuss": 14, "refer": 14}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})