# BloomFilter 
<a href="https://github.com/akulsanthosh/">
  <img src="https://contrib.rocks/image?repo=akulsanthosh/Video-Colorization" />
</a> 
<a href="https://github.com/sahildeshpande25/">
  <img src="https://contrib.rocks/image?repo=sahildeshpande25/test" />
</a>

Steps to run sequential (CPU) code
1.  ```gcc -o seq_bloomfilter -std=c99 seq_bloomfilter.c```
2.  ```./seq_bloomfilter inserts.txt lookups.txt``` 
    <br>
     
Steps to run parallel (GPU) code
1.  ```nvcc -o bloomfilter bloomfilter.cu```
2.  ```./bloomfilter inserts.txt lookups.txt``` 
     <br><br> where 
     <br>```inserts.txt``` contains the strings to be inserted
     <br> ```lookups.txt``` contains the strings to be looked up
     <br> filenames are just placeholders and can be renamed
     


Steps to generate random insert and lookup files
*   ```python3 random_input_gen.py -N <num> -P <lookup_percent> -fin <inserts_filename.txt> -flp <lookups_filename.txt>```
    <br><br> where 
    <br> ```<num>``` is the number of random strings to be inserted (default=10000)
    <br> ```<lookup_percent>``` is the percentage (0,1) of the generated inserts to be looked up (default=0.2)
    <br> The ratio of lookups to inserts is 10:1 so the remaining lookups are randomly generated 
    <br><br> ```<inserts_filename.txt>``` is the filename to store the generated inserts (default='inserts.txt')
    <br> ```<lookups_filename.txt>``` is the filename to store the generated lookups (default='lookups.txt')
    <br><br> Note: The number of generated strings for insertions and lookups is lesser than the actual value since duplicate strings are filtered out to measure performance of the bloom filter.
