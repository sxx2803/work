/******************************************************************************
* Base struct to contain data structure element information: deterimined by
* the application needs.
******************************************************************************/
#ifndef _LINKED_LISTS_H_
#define _LINKED_LISTS_H_

#include <string.h>

typedef struct ElementStructs
  {
   /* Application Specific Definitions */
    char word[255];
    int count;
  } ElementStructs;

/**************  Nothing else in the module needs to be modified *************/



/******************************************************************************
* Base struct of list nodes, contains user information and link pointers.
* The "ElementStructs" typemark must be defined based on specific needs of the
* application.
******************************************************************************/
typedef struct LinkedListNodes
  {
   /* The user information field */
   ElementStructs *ElementPtr;
   /* Link pointers */
   struct LinkedListNodes *Next;
   struct LinkedListNodes *Previous;
  } LinkedListNodes;

/******************************************************************************
* Base struct used to manage the linked list data structure.
******************************************************************************/
typedef struct LinkedLists
  {
   /* Number of elements in the list */
   int NumElements;
   /* Pointer to the front of the list of elements, possibly NULL */
   struct LinkedListNodes *FrontPtr;
   /* Pointer to the end of the list of elements, possibly NULL */
   struct LinkedListNodes *BackPtr;
  } LinkedLists;

/******************************************************************************
* Initialized the linked list data structure
******************************************************************************/
void InitLinkedList(LinkedLists *ListPtr);

/******************************************************************************
* Adds a record to the front of the list.
******************************************************************************/
void AddToFrontOfLinkedList(LinkedLists *ListPtr, ElementStructs *DataPtr);

/******************************************************************************
* Adds a record to the back of the list.
******************************************************************************/
void AddToBackOfLinkedList(LinkedLists *ListPtr, ElementStructs *DataPtr);

/******************************************************************************
* Removes (and returns) a record from the front of the list ('works' even on 
* an empty list by returning NULL).
******************************************************************************/
ElementStructs *RemoveFromFrontOfLinkedList(LinkedLists *ListPtr);

/******************************************************************************
* Removes (and returns) a record from the back of the list ('works' even on 
* an empty list by returning NULL).
******************************************************************************/
ElementStructs *RemoveFromBackOfLinkedList(LinkedLists *ListPtr);

/******************************************************************************
* De-allocates the linked list and resets the struct fields as if the
* list was empty.
******************************************************************************/
void DestroyLinkedList(LinkedLists *ListPtr);

#endif /* _LINKED_LISTS_H_ */
