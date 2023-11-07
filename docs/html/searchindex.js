Search.setIndex({"docnames": ["index", "modules", "notebooks/statistical_approach", "notebooks/tests", "tfrStats", "tfrStats.cluster_correction", "tfrStats.get_dpvals_minmax", "tfrStats.get_dpvals_whole", "tfrStats.get_pvals_minmax", "tfrStats.get_pvals_whole", "tfrStats.load_uv_tfrs", "tfrStats.plot_dtfr_stats", "tfrStats.plot_tfr_stats", "tfrStats.tfr_spw_stats_minmax", "tfrStats.tfr_spw_stats_whole", "uvtfrs"], "filenames": ["index.rst", "modules.rst", "notebooks/statistical_approach.ipynb", "notebooks/tests.ipynb", "tfrStats.rst", "tfrStats.cluster_correction.rst", "tfrStats.get_dpvals_minmax.rst", "tfrStats.get_dpvals_whole.rst", "tfrStats.get_pvals_minmax.rst", "tfrStats.get_pvals_whole.rst", "tfrStats.load_uv_tfrs.rst", "tfrStats.plot_dtfr_stats.rst", "tfrStats.plot_tfr_stats.rst", "tfrStats.tfr_spw_stats_minmax.rst", "tfrStats.tfr_spw_stats_whole.rst", "uvtfrs.md"], "titles": ["<strong>A mini-tutorial on the Statistical Assessment of Time Frequency Data</strong>", "tfrStats", "<span class=\"section-number\">2. </span>A mini-tutorial\u2026", "Import libraries and define functions", "tfrStats package", "tfrStats.cluster_correction module", "tfrStats.get_dpvals_minmax module", "tfrStats.get_dpvals_whole module", "tfrStats.get_pvals_minmax module", "tfrStats.get_pvals_whole module", "tfrStats.load_uv_tfrs module", "tfrStats.plot_dtfr_stats module", "tfrStats.plot_tfr_stats module", "tfrStats.tfr_spw_stats_minmax module", "tfrStats.tfr_spw_stats_whole module", "<span class=\"section-number\">1. </span>On TFR statistical assessments"], "terms": {"recent": [0, 3, 15], "i": [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "encount": 0, "an": 0, "articl": 0, "discuss": [0, 2], "ongo": 0, "replic": 0, "crisi": 0, "biologi": [0, 15], "1": [0, 2, 3, 10, 11, 12, 13, 14, 15], "2": [0, 2, 3, 6, 8, 10, 11, 12, 13, 14, 15], "why": 0, "so": [0, 2], "often": 0, "stress": 0, "result": [0, 2, 3, 5, 10, 11, 12], "obtain": [0, 2, 5], "differ": [0, 2, 5, 9], "team": 0, "us": [0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14], "same": [0, 2], "follow": [0, 2], "inquiri": 0, "ar": [0, 2, 3, 5, 11, 12, 13, 14, 15], "difficult": 0, "accord": 0, "scientist": 0, "tend": 0, "integr": 0, "belief": 0, "hypothesi": [0, 2], "make": [0, 2], "machineri": 0, "e": [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "form": 0, "toolbox": 0, "everi": 0, "problem": 0, "thei": [0, 11, 12], "stumbl": 0, "upon": 0, "field": [0, 15], "while": [0, 2], "need": [0, 2], "collect": 0, "consensu": 0, "clear": [0, 2], "potenti": [0, 15], "diverg": 0, "decis": 0, "taken": [0, 11, 12], "dure": [0, 2], "mai": [0, 2], "bring": 0, "forth": 0, "confus": 0, "rather": 0, "than": [0, 2], "clariti": [0, 2], "With": 0, "all": [0, 2, 3, 5], "its": [0, 5], "good": 0, "intent": 0, "excess": 0, "trust": 0, "method": [0, 2, 3, 5, 6, 7, 8, 9, 15], "should": [0, 2], "lead": 0, "realm": 0, "extrem": [0, 2, 3, 13], "scientif": 0, "scientific": 0, "scientism": 0, "urg": 0, "temporari": 0, "answer": 0, "our": 0, "ol": 0, "metric": [0, 5], "provid": [0, 2, 15], "underli": 0, "inspir": 0, "them": 0, "first": 0, "place": 0, "coincident": 0, "try": 0, "reach": [0, 2], "thi": [0, 2, 5, 6, 7, 8, 9, 10, 11, 12], "my": 0, "own": 0, "work": [0, 2, 5, 9], "anoth": 0, "noteworthi": 0, "piec": 0, "now": [0, 2, 3, 13, 14], "obsolet": 0, "twitter": 0, "The": [0, 2], "post": [0, 2], "much": 0, "zu": 0, "sagen": 0, "plumber": 0, "s": [0, 3, 15], "perspect": 0, "statistician": 0, "less": 0, "like": 0, "priest": 0, "more": [0, 2], "don": 0, "t": [0, 3, 15], "care": 0, "what": 0, "you": [0, 2], "person": 0, "believ": 0, "right": [0, 2, 3], "wai": [0, 2], "do": [0, 2], "thing": 0, "have": [0, 2, 15], "specif": [0, 2, 15], "want": [0, 2], "know": 0, "possibl": [0, 5], "solut": 0, "might": 0, "fix": 0, "limit": 0, "how": [0, 3], "each": [0, 2, 3, 5, 6, 7, 8, 9, 13], "would": [0, 2], "cost": 0, "dani\u00ebl": [0, 2], "lacken": [0, 2], "It": [0, 3, 13], "made": 0, "sens": 0, "after": 0, "plumb": 0, "fit": [0, 2], "nuanc": 0, "task": [0, 2], "can": [0, 2], "either": [0, 2], "yield": 0, "pipe": 0, "jungl": 0, "profession": 0, "design": 0, "system": 0, "In": [0, 2, 3, 13], "show": [0, 2], "two": [0, 2, 3, 10, 11, 12, 13, 14, 15], "relat": [0, 2, 15], "approach": [0, 2, 3, 13], "when": [0, 2, 3, 14], "appli": 0, "scenario": 0, "equival": [0, 2], "basic": [0, 2], "code": 0, "illustr": 0, "yet": 0, "fundament": 0, "similar": 0, "pipelin": 0, "slightli": [0, 2], "compar": 0, "base": [0, 2, 3, 5, 13, 14, 15], "exampl": [0, 2, 3, 13, 14], "fieldtrip": [0, 3, 13, 14], "adapt": [0, 5, 9], "from": [0, 2, 3, 5, 6, 7, 8, 9, 13, 14], "book": 0, "analyz": 0, "neural": [0, 15], "seri": 0, "theori": 0, "practic": 0, "anil": 0, "oza": 0, "reproduc": 0, "trial": [0, 2, 3, 13], "246": 0, "biologist": 0, "get": [0, 3, 6, 7, 8, 9], "set": 0, "natur": [0, 2, 15], "2023": [0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "botvinik": 0, "nezer": 0, "et": 0, "al": 0, "variabl": 0, "analysi": [0, 2], "singl": [0, 3, 15], "neuroimag": [0, 15], "dataset": [0, 3], "mani": [0, 2, 3], "582": 0, "84": [0, 2, 3], "88": [0, 2, 3], "2020": 0, "On": 0, "tfr": [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14], "background": 0, "open": 0, "refer": 0, "instal": 0, "comput": [0, 3, 5, 6, 7, 8, 9, 13, 14, 15], "null": [0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "distribut": [0, 3, 5, 6, 7, 8, 9, 13, 14], "threshold": [0, 5, 11, 12], "overlaid": 0, "onto": 0, "spatio": 0, "tempor": 0, "map": [0, 5, 6, 7, 8, 9], "conclus": 0, "tfrstat": [0, 2, 3], "packag": [1, 2], "submodul": 1, "cluster_correct": [1, 4, 11, 12], "modul": [1, 2, 15], "get_dpvals_minmax": [1, 4], "get_dpvals_whol": [1, 4], "get_pvals_minmax": [1, 3, 4, 11, 12], "get_pvals_whol": [1, 4, 6, 7, 8, 11, 12], "load_uv_tfr": [1, 3, 4, 11, 12], "plot_dtfr_stat": [1, 2, 3, 4], "plot_tfr_stat": [1, 2, 3, 4], "tfr_spw_stats_minmax": [1, 2, 3, 4, 10], "tfr_spw_stats_whol": [1, 2, 4], "content": 1, "assess": 2, "signific": 2, "spectral": [2, 3, 10, 11, 12, 13, 14], "estim": 2, "electrophysiolog": [2, 15], "data": [2, 3, 15], "case": 2, "lfp": 2, "we": [2, 3, 13, 14], "non": 2, "parametr": 2, "permut": [2, 3, 8, 13, 14], "test": [2, 3, 5, 14, 15], "focus": 2, "multipl": [2, 11, 12], "comparison": [2, 11, 12], "correct": [2, 3, 5, 11, 12, 13, 14], "time": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "represent": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14], "success": 2, "depend": [2, 15], "dimens": [2, 3, 14], "hand": 2, "spatial": [2, 15], "locat": [2, 15], "paramet": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "condit": [2, 3, 10, 11, 12, 13, 14], "etc": 2, "For": 2, "pedagog": 2, "purpos": 2, "here": [2, 3, 10, 11, 12, 13, 14], "focu": 2, "power": [2, 3, 10, 11, 12, 13, 14, 15], "increas": 2, "rel": 2, "baselin": [2, 3], "variant": 2, "essenti": 2, "min": [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14], "max": [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14], "which": 2, "captur": [2, 3, 13], "variat": [2, 3, 13], "whole": [2, 3, 5, 6, 9, 10, 11, 12, 13, 14], "averag": [2, 3, 5, 6, 7, 9, 14], "across": [2, 3], "sinc": 2, "sever": [2, 15], "achiev": 2, "goal": 2, "realiz": [2, 9], "other": 2, "percentil": [2, 3, 10, 11, 12, 13, 14], "directli": 2, "further": 2, "pool": [2, 3, 5, 6, 7, 8, 9, 13, 14], "accomplish": 2, "among": [2, 15], "common": 2, "veri": 2, "simpl": 2, "hope": 2, "help": 2, "those": 2, "research": [2, 15], "includ": 2, "myself": 2, "matter": 2, "touch": 2, "ground": 2, "p": [2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14], "valu": [2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14], "chang": 2, "minimum": [2, 3, 13], "maximum": [2, 3, 13], "empir": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14], "preserv": [2, 3, 14], "onc": 2, "been": [2, 15], "cumul": [2, 6, 7, 8, 9], "python": 2, "environ": 2, "jupyt": 2, "notebook": [2, 3], "support": 2, "present": 2, "progress": 2, "plan": 2, "some": 2, "refin": 2, "next": 2, "few": 2, "week": 2, "interest": 2, "emploi": 2, "ani": 2, "question": 2, "pleas": 2, "feel": 2, "free": 2, "out": 2, "me": 2, "happi": 2, "assist": 2, "download": 2, "http": [2, 15], "github": 2, "com": 2, "nicogravel": [2, 3], "To": 2, "run": 2, "clone": 2, "your": 2, "fork": 2, "local": [2, 15], "git": 2, "cd": 2, "conda": 2, "env": 2, "creat": 2, "name": 2, "dev": 2, "file": [2, 3, 10, 11, 12, 13, 14], "yml": 2, "activ": [2, 15], "pip": 2, "voil\u00e0": 2, "Then": 2, "load": [2, 3, 10], "function": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "defin": [2, 5], "path": [2, 10, 11, 12, 13, 14], "5": [2, 3, 15], "import": 2, "in_path": [2, 3], "volum": [2, 3], "out_path": [2, 3], "stat": [2, 3, 5, 11, 12], "univari": [2, 3, 15], "gandalf": [2, 3], "mwlamprop": [2, 3], "pre": 2, "coupl": 2, "examl": 2, "If": [2, 5], "wish": 2, "yourself": 2, "posess": 2, "suitabl": 2, "detail": [2, 15], "contro": 2, "hesit": 2, "scrutin": 2, "edit": 2, "within": [2, 3, 5, 15], "signal": 2, "flip": 2, "period": 2, "ha": 2, "length": 2, "unfold": 2, "6": [2, 3, 15], "n_perm": [2, 3, 13, 14], "4": [2, 3, 15], "just": 2, "see": 2, "fband": [2, 3, 7, 9, 10, 11, 12, 13, 14], "cond": [2, 3, 10, 11, 12, 13, 14], "0": [2, 3, 5, 10, 11, 12, 13, 14], "tfr_emp": [2, 3, 6, 7, 8, 9], "tfr_null": [2, 3, 6, 7, 8, 9], "site": [2, 3, 13], "indic": [2, 3], "12": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "24": [2, 3], "36": [2, 3], "48": [2, 3], "60": [2, 3, 15], "72": [2, 3], "96": [2, 3], "108": [2, 3], "13": [2, 3, 15], "25": [2, 3, 15], "37": [2, 3], "49": [2, 3], "61": [2, 3], "73": [2, 3], "85": [2, 3, 15], "97": [2, 3], "109": [2, 3], "14": [2, 3, 15], "26": [2, 3], "38": [2, 3], "50": [2, 3], "62": [2, 3], "74": [2, 3], "86": [2, 3], "98": [2, 3], "110": [2, 3], "3": [2, 3, 10, 11, 12, 13, 14, 15], "15": [2, 3, 15], "27": [2, 3], "39": [2, 3], "51": [2, 3], "63": [2, 3], "75": [2, 3], "87": [2, 3], "99": [2, 3], "111": [2, 3, 15], "16": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "28": [2, 3, 15], "40": [2, 3, 15], "52": [2, 3], "64": [2, 3], "76": [2, 3], "100": [2, 3, 10, 11, 12, 13, 14, 15], "112": [2, 3], "17": [2, 3], "29": [2, 3, 15], "41": [2, 3], "53": [2, 3], "65": [2, 3], "77": [2, 3], "89": [2, 3, 15], "101": [2, 3], "113": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14], "18": [2, 3], "30": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14], "42": [2, 3], "54": [2, 3], "66": [2, 3], "78": [2, 3], "90": [2, 3], "102": [2, 3], "114": [2, 3], "7": [2, 3, 15], "19": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "31": [2, 3, 15], "43": [2, 3, 15], "55": [2, 3], "67": [2, 3], "79": [2, 3], "91": [2, 3], "103": [2, 3], "115": [2, 3], "8": [2, 3, 15], "20": [2, 3, 15], "32": [2, 3], "44": [2, 3], "56": [2, 3], "68": [2, 3], "80": [2, 3], "92": [2, 3], "104": [2, 3], "116": [2, 3], "9": [2, 3, 15], "21": [2, 3], "33": [2, 3], "45": [2, 3, 15], "57": [2, 3], "69": [2, 3], "81": [2, 3], "93": [2, 3, 15], "105": [2, 3], "117": [2, 3], "10": [2, 3, 15], "22": [2, 3], "34": [2, 3], "46": [2, 3], "58": [2, 3], "70": [2, 3], "82": [2, 3], "94": [2, 3], "106": [2, 3], "118": [2, 3], "11": [2, 3, 15], "23": [2, 3], "35": [2, 3], "47": [2, 3], "59": [2, 3], "71": [2, 3], "83": [2, 3], "95": [2, 3], "107": [2, 3], "119": [2, 3], "ftpool_grat_high_wavelet": [2, 3], "mat": [2, 3, 13, 14], "120": [2, 3], "uvtfr_stats_high_grat_spw_4": 2, "npz": [2, 3, 10, 11, 12, 13, 14], "uvtfr_stats_high_grat_spw_4_minmax": 2, "plot": [2, 11, 12], "val": [2, 5], "05": [2, 3, 5, 15], "top": 2, "panel": 2, "blue": 2, "trace": 2, "alpha": [2, 3, 5, 11, 12, 13, 14, 15], "mask": 2, "specifi": 2, "section": 2, "contain": [2, 3, 13, 14, 15], "abov": [2, 11, 12], "cutoff": [2, 3], "bottom": 2, "red": 2, "space": [2, 5, 6, 7, 8, 9], "one": 2, "must": 2, "consid": 2, "over": 2, "predefin": 2, "bin": [2, 3, 5, 6, 7, 8, 9, 11, 12], "well": [2, 11, 12], "diment": 2, "roi": 2, "therefor": 2, "save": [2, 3], "minim": 2, "maxim": 2, "iter": 2, "type": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "minmax": 2, "uvtfr_stats_high_grat_spw_1000_minmax": 2, "emp": [2, 3], "1000": [2, 3, 6, 8, 10, 11, 12, 13, 14], "96054268486813": 2, "peak": 2, "rang": [2, 3], "6390137791448": 2, "91082666125475": 2, "relev": 2, "uvtfr_stats_high_grat_spw_100": 2, "79162686024465": 2, "52901730660238": 2, "level": 2, "enough": 2, "between": [2, 5], "howev": 2, "special": 2, "account": [2, 3, 11, 12, 13, 14], "As": 2, "consequ": [2, 15], "absent": 2, "landcap": 2, "explan": 2, "take": 2, "procedur": 2, "give": 2, "rise": 2, "bias": 2, "experi": 2, "wherea": 2, "appproach": 2, "doe": 2, "partial": 2, "seem": 2, "larg": 2, "difer": 2, "truncat": [2, 3, 13], "Of": 2, "cours": 2, "choos": 2, "turn": 2, "landscap": 2, "size": [2, 3, 5, 11, 12, 13, 14, 15], "come": 2, "complex": 2, "scrutini": 2, "corrobor": 2, "numpi": 3, "np": 3, "secret": 3, "arang": 3, "idx": 3, "a_": 3, "roll": 3, "randbelow": 3, "print": 3, "tqdm": 3, "auto": 3, "scipi": 3, "io": 3, "sio": 3, "inf": 3, "def": 3, "tfr_spw_stats_whole_rol": 3, "svar": [3, 10, 11, 12, 13, 14], "statist": [3, 5, 6, 7, 8, 9, 13, 14], "asess": [3, 13, 14], "keep": [3, 13, 14], "todo": 3, "implement": [3, 10, 11, 12, 13, 14], "onset": [3, 13, 14], "shift": [3, 13, 14, 15], "triakl": [3, 14], "current": [3, 13, 14, 15], "400": [3, 13, 14], "ms": [3, 13, 14], "window": [3, 13, 14], "compatibilityu": [3, 13, 14], "syncopi": [3, 13, 14], "reli": [3, 13, 14], "ftpool_": [3, 13, 14], "param": 3, "string": [3, 10, 11, 12, 13, 14], "input_path": [3, 10, 11, 12, 13, 14], "index": [3, 10, 11, 12, 13, 14], "int": [3, 10, 11, 12, 13, 14], "gpr": [3, 10, 11, 12, 13, 14], "frequenc": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "band": [3, 10, 11, 12, 13, 14, 15], "low": [3, 10, 11, 12, 13, 14], "high": [3, 10, 11, 12, 13, 14, 15], "higher": [3, 10, 11, 12, 13, 14], "ob": [3, 10, 11, 12, 13, 14], "nulltyp": [3, 10, 11, 12, 13, 14], "integer": [3, 10, 11, 12, 13, 14], "cluster": [3, 5, 11, 12, 13, 14, 15], "cluster_s": [3, 5, 11, 12, 13, 14], "float": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "return": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "n_cond": [3, 10, 11, 12, 13, 14], "x": [3, 10, 11, 12, 13, 14], "n_site": [3, 10, 11, 12, 13, 14], "n_freq": [3, 10, 11, 12, 13, 14], "n_time": [3, 10, 11, 12, 13, 14], "rtype": 3, "author": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "nicola": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "gravel": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "09": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "tp": 3, "141": 3, "140": 3, "fp": 3, "block": 3, "grat": 3, "nat": 3, "spw": 3, "han": 3, "wavelet": 3, "n_sess": 3, "els": 3, "organ": 3, "channel": [3, 15], "total": 3, "site_idx": 3, "zero": 3, "astyp": 3, "uint": 3, "n": [3, 9], "bs_t0": 3, "700": 3, "bs_t1": 3, "elif": 3, "fname": 3, "str": 3, "_": 3, "loadmat": 3, "datapool": 3, "datalump_": 3, "shape": [3, 15], "linspac": 3, "start": 3, "800": 3, "stop": 3, "2000": 3, "num": 3, "b0": 3, "searchsort": 3, "side": 3, "left": 3, "sorter": 3, "none": 3, "bf": 3, "tfr_": 3, "i_cond": 3, "i_rep": 3, "i_depth": 3, "i_freq": 3, "nanmean": 3, "axi": 3, "session": 3, "x_b": 3, "flatten": 3, "nan": 3, "repetit": 3, "t0": 3, "tf": 3, "x_h0": 3, "msg": 3, "choic": 3, "random": 3, "desc": 3, "posit": 3, "i_perm": 3, "true": [3, 5], "xx_b": 3, "tile": 3, "fals": 3, "uvtfr_stats_": 3, "_roll": 3, "savez": 3, "tfr_spw_stats_minmax_rol": 3, "record": [3, 13], "ditribut": [3, 13], "200": [3, 15], "1200": 3, "win": 3, "1000m": 3, "depth": [3, 11], "nanmin": 3, "nanmax": 3, "_minmax_rol": 3, "var": 3, "folder": 3, "_n": 3, "cg8c3_pj1_778vx80m_y0nww0000gn": 3, "ipykernel_3932": 3, "1636253498": 3, "py": 3, "runtimewarn": 3, "mean": 3, "empti": 3, "slice": 3, "144": 3, "134": 3, "152": 3, "uvtfr_stats_high_grat_spw_30_rol": 3, "3094451537": 3, "262": 3, "264": 3, "270": 3, "303": 3, "293": 3, "uvtfr_stats_high_grat_spw_100_minmax_rol": 3, "whole_rol": 3, "30031964473675": 3, "minmax_rol": 3, "indexerror": 3, "traceback": 3, "most": [3, 15], "call": 3, "last": 3, "user": 3, "document": 3, "websit": 3, "tfrstats_loc": 3, "doc": 3, "sourc": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "ipynb": 3, "cell": 3, "line": 3, "lt": 3, "href": 3, "vscode": 3, "x23szmlszq": 3, "3d": [3, 5], "gt": 3, "tail": [3, 6, 8], "nulldist": 3, "both": 3, "amax": 3, "too": 3, "arrai": [3, 5, 9], "dimension": [3, 9], "were": 3, "nearest": 5, "neighbour": [5, 11, 12], "l1": 5, "unweight": 5, "distanc": [5, 11, 12], "pvalu": 5, "below": 5, "center": 5, "assign": 5, "2d": 5, "add": [5, 9], "un": 5, "g": [5, 15], "merg": [6, 7, 8], "null_tfr": [6, 7, 8, 9], "nul": [7, 8, 9], "option": [9, 11, 12], "handl": [10, 11, 12], "dictionari": [10, 11, 12], "also": [11, 12], "alreadi": [11, 12], "alltoghet": [11, 12], "suggest": 15, "brain": 15, "hold": 15, "stimulu": 15, "inform": 15, "visual": 15, "cortex": 15, "evalu": 15, "recommend": 15, "georgio": 15, "michalarea": 15, "julien": 15, "vezoli": 15, "stan": 15, "van": 15, "pelt": 15, "jan": 15, "mathij": 15, "schoffelen": 15, "henri": 15, "kennedi": 15, "pascal": 15, "fri": 15, "beta": 15, "gamma": 15, "rhythm": 15, "subserv": 15, "feedback": 15, "feedforward": 15, "influenc": 15, "human": 15, "cortic": 15, "area": 15, "neuron": 15, "384": 15, "397": 15, "2016": 15, "doi": 15, "1016": 15, "j": 15, "2015": 15, "018": 15, "christoph": 15, "m": 15, "lewi": 15, "conrado": 15, "A": 15, "bosman": 15, "brunet": 15, "bruss": 15, "lima": 15, "mark": 15, "robert": 15, "thilo": 15, "womelsdorf": 15, "peter": 15, "de": 15, "weerd": 15, "sergio": 15, "neuenschwand": 15, "wolf": 15, "singer": 15, "biorxiv": 15, "1101": 15, "049718": 15, "supratim": 15, "rai": 15, "nathan": 15, "crone": 15, "ernst": 15, "niebur": 15, "piotr": 15, "franaszczuk": 15, "steven": 15, "hsiao": 15, "correl": 15, "oscil": 15, "hz": 15, "macaqu": 15, "implic": 15, "electrocorticographi": 15, "journal": 15, "neurosci": 15, "11526": 15, "11536": 15, "2008": 15, "hiroya": 15, "ono": 15, "masaki": 15, "sonoda": 15, "brian": 15, "h": 15, "silverstein": 15, "kaori": 15, "takafumi": 15, "kubota": 15, "aime": 15, "f": 15, "luat": 15, "rothermel": 15, "sandeep": 15, "sood": 15, "eishi": 15, "asano": 15, "spontan": 15, "clinic": 15, "neurophysiolog": 15, "132": 15, "2391": 15, "2403": 15, "2021": 15, "xiaoxuan": 15, "jia": 15, "matthew": 15, "smith": 15, "adam": 15, "kohn": 15, "select": 15, "coher": 15, "compon": 15, "9390": 15, "9403": 15, "2011": 15, "jeremi": 15, "r": 15, "man": 15, "joshua": 15, "jacob": 15, "itzhak": 15, "michael": 15, "kahana": 15, "broadband": 15, "spectra": 15, "spike": 15, "13613": 15, "13620": 15, "2009": 15, "1523": 15, "jneurosci": 15, "2041": 15, "d": 15, "herm": 15, "k": 15, "miller": 15, "b": 15, "wandel": 15, "winaw": 15, "cerebr": 15, "2951": 15, "2959": 15, "2014": 15, "1093": 15, "cercor": 15, "bhu091": 15, "reflect": 15, "imag": 15, "structur": 15, "635": 15, "643": 15, "2019": 15, "04": 15, "440025": 15, "timo": 15, "kerkoerl": 15, "w": 15, "self": 15, "bruno": 15, "dagnino": 15, "mari": 15, "alic": 15, "gariel": 15, "mathi": 15, "jasper": 15, "poort": 15, "chri": 15, "der": 15, "togt": 15, "pieter": 15, "roelfsema": 15, "character": 15, "process": 15, "monkei": 15, "proceed": 15, "nation": 15, "academi": 15, "scienc": 15, "14332": 15, "14341": 15, "1073": 15, "pna": 15, "1402773111": 15, "maryam": 15, "bijanzadeh": 15, "lauri": 15, "nurminen": 15, "sam": 15, "merlin": 15, "andrew": 15, "clark": 15, "alessandra": 15, "angelucci": 15, "distinct": 15, "laminar": 15, "global": 15, "context": 15, "primat": 15, "primari": 15, "259": 15, "274": 15, "e4": 15, "2018": 15, "08": 15, "020": 15, "lar": 15, "muckli": 15, "federico": 15, "martino": 15, "luca": 15, "vizioli": 15, "luci": 15, "petro": 15, "fraser": 15, "kamil": 15, "ugurbil": 15, "rainer": 15, "goebel": 15, "essa": 15, "yacoub": 15, "contextu": 15, "superfici": 15, "layer": 15, "v1": 15, "2690": 15, "2695": 15, "cub": 15, "057": 15, "andr\u00e9": 15, "mora": 15, "basto": 15, "arturo": 15, "oostenveld": 15, "jarrod": 15, "dowdal": 15, "exert": 15, "through": 15, "390": 15, "401": 15, "benjamin": 15, "fischer": 15, "detlef": 15, "wegen": 15, "epidur": 15, "about": 15, "color": 15, "commun": 15, "690": 15, "1038": 15, "s42003": 15, "021": 15, "02207": 15, "eric": 15, "nonparametr": 15, "eeg": 15, "meg": 15, "164": 15, "177": 15, "190": 15, "2007": 15, "jneumeth": 15, "03": 15, "024": 15, "c": 15, "pernet": 15, "latinu": 15, "nichol": 15, "rousselet": 15, "mass": 15, "analys": 15, "event": 15, "potentialsield": 15, "250": 15, "cut": 15, "edg": 15, "org": 15, "003": 15}, "objects": {"": [[4, 0, 0, "-", "tfrStats"]], "tfrStats": [[5, 0, 0, "-", "cluster_correction"], [6, 0, 0, "-", "get_dpvals_minmax"], [7, 0, 0, "-", "get_dpvals_whole"], [8, 0, 0, "-", "get_pvals_minmax"], [9, 0, 0, "-", "get_pvals_whole"], [10, 0, 0, "-", "load_uv_tfrs"], [11, 0, 0, "-", "plot_dtfr_stats"], [12, 0, 0, "-", "plot_tfr_stats"], [13, 0, 0, "-", "tfr_spw_stats_minmax"], [14, 0, 0, "-", "tfr_spw_stats_whole"]], "tfrStats.cluster_correction": [[5, 1, 1, "", "cluster_correction"]], "tfrStats.get_dpvals_minmax": [[6, 1, 1, "", "get_dpvals_minmax"]], "tfrStats.get_dpvals_whole": [[7, 1, 1, "", "get_dpvals_whole"]], "tfrStats.get_pvals_minmax": [[8, 1, 1, "", "get_pvals_minmax"]], "tfrStats.get_pvals_whole": [[9, 1, 1, "", "get_pvals_whole"]], "tfrStats.load_uv_tfrs": [[10, 1, 1, "", "load_uv_tfrs"]], "tfrStats.plot_dtfr_stats": [[11, 1, 1, "", "plot_dtfr_stats"]], "tfrStats.plot_tfr_stats": [[12, 1, 1, "", "plot_tfr_stats"]], "tfrStats.tfr_spw_stats_minmax": [[13, 1, 1, "", "tfr_spw_stats_minmax"]], "tfrStats.tfr_spw_stats_whole": [[14, 1, 1, "", "tfr_spw_stats_whole"]]}, "objtypes": {"0": "py:module", "1": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "function", "Python function"]}, "titleterms": {"A": [0, 2], "mini": [0, 2], "tutori": [0, 2], "statist": [0, 2, 15], "assess": [0, 15], "time": 0, "frequenc": [0, 2], "data": 0, "content": [0, 4], "python": 0, "packag": [0, 4], "tfrstat": [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "instal": 2, "comput": 2, "null": 2, "distribut": 2, "threshold": 2, "overlaid": 2, "onto": 2, "spatio": 2, "tempor": 2, "map": 2, "conclus": 2, "import": 3, "librari": 3, "defin": 3, "function": 3, "set": 3, "path": 3, "submodul": 4, "modul": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "cluster_correct": 5, "todo": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "get_dpvals_minmax": 6, "get_dpvals_whol": 7, "get_pvals_minmax": 8, "get_pvals_whol": 9, "load_uv_tfr": 10, "plot_dtfr_stat": 11, "plot_tfr_stat": 12, "tfr_spw_stats_minmax": 13, "tfr_spw_stats_whol": 14, "On": 15, "tfr": 15, "background": 15, "open": 15, "discuss": 15, "refer": 15}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})