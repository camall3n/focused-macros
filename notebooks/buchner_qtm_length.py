from domains.cube import pattern

for start in range(10,20):
    a = [len(pattern.buchner2018pattern(seed=s)) for s in range(start*10, (start+1)*10)]
    print(start*10, sum(a)/len(a))
print()
a = [len(pattern.buchner2018pattern(seed=s)) for s in range(100, 200)]
print('min ', min(a))
print('mean', sum(a)/len(a))
print('max ', max(a))
