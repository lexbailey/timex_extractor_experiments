#!/usr/bin/env python
from timex_finder_bayes import export_table
import json

tables = export_table()

data = {
i:table.probs
for i, table in tables.items()
}

print("""
from bayes_extractor import ProbabilityTable
def get_tables():
    tables = {}
""")
for i, table in data.items():
    #print("tables[%d] = ProbabilityTable.from_json(%r)" % (i, table))
    print("    tables[%d] = ProbabilityTable()" % (i))
    print("    tables[%d].probs = %r" % (i, table))

print("""
    return tables
""")
