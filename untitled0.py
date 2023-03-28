# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:59:56 2023

@author: Avell
"""

# n = 955

# s = [0] * 30
# q = 0
# # Transform to binary
# while (n > 0):
#     s[q] = n % 2
#     n //= 2
#     q += 1

# # 
# for p in range(0, q//2 + 1):
#     ok = True
#     for k in range(q - p):
#         if s[k] != s[k + p]:
#             ok = False
#             break
#     if ok:
#         print(p)
        
import json

class SearchByTag:
    def __init__(self, data_file, query_tag):
        with open(data_file) as data_file:
            self._data = json.load(data_file)
            # self._data = data.items
        self.query = query_tag

    def search(self):
        for key, value in self._data.items():
            total = len(value)
            n = 0
            while n <= total:
                item = dict(value[n])
                tags = set(item["tags"])
                for tag in tags:
                    print(tag)
                    if (tag == self.query):
                        print("AQUI")
                        yield value
                n += 1

    def first(self):
        s = self.search()
           
        # raise StopIteration


search = SearchByTag("movies.json", "action")

searchs = search.search()
print(next(searchs))     
search.first()

# A = [1,10,10,40,50,60]

# prev, diff = -1, 10 ** 9
# for a in A:
#     print("a = ", a)
#     print("prev = ", prev)
#     if prev != -1:
#         diff = min(diff, abs(a - prev))
#         print(diff)
#     prev = a