class Node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
root = None

def createtree(root):
    print("Enter the value")
    val = int(input())
    if val == 0:
        return
    if root == None:
        root = Node(val)
       
    elif root.left == None:
        root.left = Node(val)
    elif root.right == None:
        root.right = Node(val)
    createtree(root.left)
    return
    createtree(root.right)
    return root

root = createtree(root)
def dfs(root):
    if root==None:
        print("root is none")
        return
    print(root.value)
    dfs(root.left)
    dfs(root.right)
dfs(root)





        
