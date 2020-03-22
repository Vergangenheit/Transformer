import pipeline as pp
import positional_embedding as pe

data_en, data_fr_in, dataset = pp.pipeline()
pes = pe.build_pes(data_en, data_fr_in)
