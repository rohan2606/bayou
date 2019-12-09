import os

# give offset the number you want to start with - 1

offset = 67
for a in os.listdir():
     if 'Program_output' in a:
         number = int(a.split('.')[0].split('_')[-1])
         number += offset 
         replace_with = 'Program_output_' + str(number) + '.json' 
         os.rename(a, replace_with)
         print(f'Replacing {a} with {replace_with}' )
