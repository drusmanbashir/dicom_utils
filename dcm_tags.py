import ipdb
from helpers import *
tr = ipdb.set_trace

no_change = lambda inpt:  inpt
st = lambda inpt: 'st'+inpt.replace(".","")
def human_readable_desc(text):
    unwanted_chars = ["/", "\\", ".", " ","+"]
    for ch in unwanted_chars:
        if ch in text:
            text = text.replace(ch, "")
    return text.lower()

_TRANSLATORS = {
    'StudyDate': no_change, 
    'SliceThickness': st,
    'StudyDescription':human_readable_desc
}
_PLACEHOLDERS = {
   
    'StudyDate': 'nodate',
    'SliceThickness': '',
    'StudyDescription': 'nodesc'
}

@process_attr
def translate_tag(tag,val):
    val = str(val)
    if len(val)>0:
        if tag in _TRANSLATORS:
            return  _TRANSLATORS[tag](val)
        else: return  val
    else: return _PLACEHOLDERS[tag]
    


