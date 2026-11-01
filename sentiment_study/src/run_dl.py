import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import seq_models as S
t0 = time.time()
mk = sys.argv[1]
if mk == "IN":
    S.run("IN", ["up_next", "big_next"], ["T", "T+S", "T+G", "T+G+S"], seeds=(0, 1, 2), block=3)
    S.run("IN", ["up_today"], ["T", "T+S", "T+G", "T+G+S"], seeds=(0, 1, 2), block=3)
else:
    S.run("US", ["up_next", "big_next"], ["T", "T+S", "T+G", "T+G+S"], seeds=(0, 1, 2), block=4)
    S.run("US", ["up_today"], ["T", "T+S", "T+G", "T+G+S"], seeds=(0, 1, 2), block=4)
print("DL DONE", mk, round(time.time() - t0), flush=True)
