# BloomFilter

Steps to run sequential code
1.  ```gcc -o bloomfilter -std=c99 bloomfilter.c```
2.  ```./bloomfilter inserts.txt lookups.txt``` 
     <br><br> where 
     <br>```inserts.txt``` contains the strings to be inserted
     <br> ```lookups.txt``` contains the strings to be looked up
     <br> filenames are just placeholders and can be renamed
     


Steps to generate random insert and lookup files
*   ```python3 random_input_gen.py -N <num> -P <lookup_percent> -fin <inserts_filename.txt> -flp <lookups_filename.txt>```
    <br><br> where 
    <br> ```<num>``` is the number of random strings to be inserted
    <br> ```<lookup_percent>``` is the percentage (0,1) of the generated inserts to be looked up
    <br> The ratio of lookups to inserts is 10:1 so the remaining lookups are randomly generated
    <br><br> ```<inserts_filename.txt>``` is the filename to store the generated inserts
    <br> ```<lookups_filename.txt>``` is the filename to store the generated lookups
