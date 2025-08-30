n=int(input('enter no of subj'))
total=0

for i in range(n):
    marks=float(input(f"enter a marks for subj{i+1}: "))
    total+=marks

avg=total/n
    
if avg>=90:
    grade="A+"
elif avg>=80:
    grade="A"
elif avg>=70:
    grade="B+"
elif avg>=60:
    grade="B"
elif avg>=50:
    grade="C+"
elif avg>=40:
    grade="C"



print("\n==Report Card==")
print(f'Average Marks:{avg:.2f}%')
print(f"final grade: {grade}")














    
