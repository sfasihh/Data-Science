from lxml import etree
from lxml.etree import XMLSyntaxError
import sys
import os

fn = 'dblp.xml'
for event, elem in etree.iterparse(fn, load_dtd=True):
    if elem.tag not in ['article', 'inproceedings', 'proceedings']:
        continue

    title = elem.find('title') 
    author = elem.find('author')
    
    if author is not None:

        writepath = '/Users/samiya/Desktop/collection/'
        writepath = writepath + author.text   
        text = title.text
        try:  
            f=open(writepath, "a+")
            if (text is not None):
                f.write("{}\n".format(text  ))
        except:
            print("Could not write to file: " + writepath) 
        finally:
            f.close()
            
    elem.clear()