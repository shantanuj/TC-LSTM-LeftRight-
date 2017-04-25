 

'''
Works for semeval 14' task 5
Output csv as sentence, aspect word, polarity

For sentence with 2 aspects make two rows 

Ignore sentences with no aspects, store later for manual labelling

'''

import xml.etree.ElementTree as ET 
import os
import csv 
cdir=os.getcwd()
path_to_xml=cdir+"/xmldata/Laptops_Test_Gold.xml" #feel free to change

tree=ET.parse(path_to_xml)
output_csv_name_pre=path_to_xml.split('/')[-1]
output_csv_name=output_csv_name_pre[:output_csv_name_pre.index('.xml')] #feel free to set custom
root=tree.getroot()
sentences=""
aspectTerm=""
aspectPolarity=""
output_count=1

output_list=[]#[[count,sentence,aspectTerm,aspectPolarity]]
no_aspect_list=[] #[count,sentence] <-- count is used as index
no_aspect_count=0 #this is just for counting, not used to index 
for i in range(len(root)):
	sentence=root[i][0].text #sentence(i)>text.text
	#root[i].attrib gives sentence id incase required
	if(len(root[i])>1): #check if aspect terms exist
		
		for j in range(len(root[i][1])):
			#aspect_start=root[i][1][j].attrib['from'] #AspectTerms>aspectTerm
			#aspect_end=root[i][1][j].attrib['to']
			aspect_term=root[i][1][j].attrib['term']
			aspect_polarity=root[i][1][j].attrib['polarity']
			
			output_list.append([sentence,aspect_term,aspect_polarity])
			output_count+=1
	else:
		no_aspect_list.append([output_count,sentence])
		no_aspect_count+=1
		output_count+=1


with open(cdir+"/datasets/"+output_csv_name+".csv","a") as f:
	writer=csv.writer(f,delimiter=";") #delimiter is ;
	writer.writerows(output_list)
with open(cdir+"/datasets/"+output_csv_name+"_no_aspects.csv",'a') as f:
	writer=csv.writer(f,delimiter=";")
	writer.writerows(no_aspect_list)

print("Total number of sentences: %d"%(output_count))
print("Number of non aspect sentences: %d"%(no_aspect_count))
