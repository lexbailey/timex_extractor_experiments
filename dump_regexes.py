#!/usr/bin/env python
from timex_finder_regex import timex_res, anti_timex_res
print("import re")
print("timex_res = []")
for re_list in timex_res:
    print("timex_res.append([\n\t")
    print("\n\t,".join("re.compile(%r)" % (re.pattern) for re in re_list))
    print("])")
print("anti_timex_res = []")
for re in anti_timex_res:
    print("anti_timex_res.append(%r)" % (re))
