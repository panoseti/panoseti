import redis

r = redis.Redis(host='localhost', port=6379, db=0)

f = open("modulePair.config")

lines = f.readlines()

for line in lines:
    if line[0] != '#':
        for i in line.split():
            boardLoc = (int(i) << 2) & 0xfffc
            for i in range(4):
                r.hset('UPDATED', boardLoc, 0)
                print(boardLoc, ' SETUP')
                boardLoc+= 1

f.close()
