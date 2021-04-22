model = []
for c in corpus:
    model.append(lda[c])

t = AnnoyIndex(20, 'angular')

counter = -1
for i in model:
    counter += 1
    test = i[0]
    v = []
    
    for c in range(20):
        v.append(0)
        
    for c in test:   
        v[c[0]] = c[1]
    
    t.add_item(counter, v)
        
t.build(1000)

path = 'C:\\Users\\samiy\\Desktop\\sam\\collection'

while True:

    author_input = input("Author Name: ")
    
    item = author_name.index(author_input + ".txt")
    
    similarity = t.get_nns_by_item(item, 10)
    
    print("\n" + author_name[item] + "'s Topic Distribution:")
    print(model[item])
    
    print("\nTen authors similar to " + author_input + ":")
        
    for i in similarity:
        print(author_name[i])
		#print("\nTopic Distribution: \n")
        print(model[i])