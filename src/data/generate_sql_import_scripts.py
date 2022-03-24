import glob

# the directory that contains directories like "data_Q1_2021"
directory = "./"
file_read = glob.glob(directory+"*/*")
print(file_read[0])

with open("sql_import", mode="w", encoding="utf-8") as file:
    things_before_data = "before"
    file.write(".mode csv")
    file.write(".echo on")
    for item in file_read:
        line = ".import " + item + " drive_stats"
        #print(line)
        file.write(item)
    file.write("delete from drive_stats where model = 'model';")
