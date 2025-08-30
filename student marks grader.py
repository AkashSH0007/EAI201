def grade(percentage):   
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"

name = input("Enter student name: ")
num_subjects = int(input("Enter number of subjects: "))
Marks=int(input("Enter max marks of subject"))
total_marks = 0
total_max = num_subjects * Marks  

for i in range(num_subjects):          
    marks = float(input(f"Enter marks for Subject {i+1}: "))
    total_marks += marks

percentage = (total_marks / total_max) * 100  
grade =grade(percentage)

print("\n----- Student Report -----")
print(f"Name       : {name}")
print(f"Total Marks: {total_marks}/{total_max}")
print(f"Percentage : {percentage:.2f}%")
print(f"Grade      : {grade}")
print("--------------------------")

