```
Tuesday October 17th

Lara Pesa, Chief of Staff 1:30-2
Milton Li, Technical Architect 2-2:30
Semih Sogutlu, Engineering Lead 2:30-3
William Ward Data Engineer 3-3:30
Sruthi Machina, Data Scientist 3:30-4
Satya Raje & Amarachi Miller Vp of Data & Vp of Product 4-4:30
```

```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head: return None
        
        odd = head
        evenhead = even = odd.next

        while even and even.next:
            odd.next = odd.next.next
            odd = odd.next
            even.next = even.next.next
            even = even.next
        odd.next = evenhead
        return head
                 
                 
```

