import os
import sys
import string
import random
import argparse

lookups = list()
inserts = list()
lookups_set = list()

class Constants:
	
	characters = list(string.ascii_letters + string.digits)
	random.shuffle(characters)
	min_len = 1
	max_len = 8
	lookup_insert_ratio = 10

def gen_random_string():
	
	'''generates a string of alphanumeric values'''

	str_length = random.randrange(Constants.min_len, Constants.max_len)
	s = []
	for j in range(str_length):
		s.append(random.choice(Constants.characters))

	return ''.join(s)

def gen_samples(n, lookup_frequency, insert_filename):
	
	'''generates n random strings each of variable length'''

	with open(insert_filename, 'w') as f:
		global inserts
		# f.write(str(n) + '\n')
		
		for i in range(n):

			s = gen_random_string()
			# f.write(s + '\n')
			inserts.append(s + '\n')

			if i%lookup_frequency == 0:
				lookups.append(s + '\n')

		inserts = list(set(inserts))
		n = len(inserts)
		f.write(str(n) + '\n')

		for i in inserts:
			f.write(i)


def gen_lookups(n, lookup_filename):
	
	''' generates n lookups'''
	

	pos = 0
	
	with open(lookup_filename, 'w') as f:
		global lookups_set
		# f.write(str(n) + '\n')

		for i in range(n):

			if (pos < len(lookups) and random.uniform(0, 1) > 0.5):
				s = lookups[pos]
				pos += 1
				lookups_set.append(s)

			else:
				s = gen_random_string()
				lookups_set.append(s + '\n')


			# f.write(s + '\n')

		lookups_set = list(set(lookups_set))
		n = len(lookups_set)
		f.write(str(n) + '\n')

		for i in lookups_set:
			f.write(i)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument('-N', type=int, default=10000,
		help='number of elements to be inserted (default=10000)')
	parser.add_argument('-P', type=float, default=0.2, 
		help='percentage (0, 1) of inserted elements to lookup (rest are random)')
	parser.add_argument('-fin', type=str, default='inserts.txt',
		help='filename to store generated strings for inserts')
	parser.add_argument('-flp', type=str, default='lookups.txt',
		help='filename to store generated strings for lookups')

	args = parser.parse_args()

	n_inserts = args.N
	n_lookups = n_inserts * Constants.lookup_insert_ratio
	lookup_percentage = args.P
	insert_filename = args.fin
	lookup_filename = args.flp

	if not (0.0 < lookup_percentage < 1.0):
		sys.exit('Lookup percentage must be in range (0, 1)')
		

	lpf = int(1/lookup_percentage)

	gen_samples(n_inserts, lpf, insert_filename)
	gen_lookups(n_lookups, lookup_filename)