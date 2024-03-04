mylist = []

def add_to_list(to_add, my_list) -> None:
    my_list.append(to_add)

def add_text_to_list(my_list) -> None:
    my_list.append(input("write, what you have to say:\n"))

def search_for_string(my_list, search_string: str) -> int: #returnes the position of the first instance of a string in the list, returnes -1 if none found
    for i in range(len(my_list)): # i know that i can run through the list itself, but then i would have to add a integer wich gives me the position anyway
        if mylist[i] == search_string:
            return i
    return -1

# just some calls to enshure the programm works
add_to_list("foo", mylist)
add_text_to_list(mylist)
print(search_for_string(mylist, "foo"))
print(search_for_string(mylist, "fool"))
