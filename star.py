
for y in range(21):
	for x in range(21):
		r = (x - 10)*(x - 10) + (y - 10)*(y - 10)
		if( r < 81 and r >= 36):
			print("* ", end = '')
		else:
			print("  ", end = '')
	print("")

print("\n\n")

for y in range(6,20):
	print("        ", end = '')
	for x in range(1,15):
		if((x<10 and y>10 ) or ( y+x < 30 and y+x > 10 and  x + y != 20 and y !=10 and x!= 10) or (y == 10 and x > 10) or (x == 10 and y < 10)):
			print("* ", end = '')
		else:
			print("  ", end = '')
	print("")


print("\n\n")