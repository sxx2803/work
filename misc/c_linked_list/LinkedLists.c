/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 3,                                                      */ 
/*                                                                  */ 
/* Linked Lists - This program represents a linked list structure   */
/********************************************************************/

#include <stdlib.h>
#include "LinkedLists.h"

/* Initializes a linked list structure */
void InitLinkedList(LinkedLists *ListPtr){
	/* Initialize number of elements to 0 */
	ListPtr->NumElements = 0;
	/* Set front element to null */
	ListPtr->FrontPtr = NULL;
	/* Set back element to null */
	ListPtr->BackPtr = NULL;
}

/* Adds a record to the front of the list */
void AddToFrontOfLinkedList(LinkedLists *ListPtr, ElementStructs *DataPtr){

	/* Variables to hold current and new node */
	LinkedListNodes *newNode;
	LinkedListNodes *curNode;
	
	/* Empty linked list */
	if(ListPtr->NumElements < 1){
		/* Create a new node to be added */
		newNode = (LinkedListNodes*) malloc(sizeof(LinkedListNodes));
		newNode->ElementPtr = DataPtr;

		/* Set node neighbors to be null because of empty list */
		newNode->Previous = NULL;
		newNode->Next = NULL;

		/* Set up the pointers of the linked list to the node */
		ListPtr->FrontPtr = newNode;
		ListPtr->BackPtr = newNode;

		/* Increment the number of elements */
		ListPtr->NumElements += 1;
	}

	/* All other list sizes */
	else{
		/* Create a new node to be added */
		newNode = (LinkedListNodes*) malloc(sizeof(LinkedListNodes));
		newNode->ElementPtr = DataPtr;

		/* Get the current node at the front of the list */
		curNode = ListPtr->FrontPtr;

		/* Set up pointers in the new data node */
		newNode->Previous = NULL;
		newNode->Next = curNode;

		/* Set up pointers in the current head */
		curNode->Previous = newNode;

		/* Set up pointers for managing structure */
		ListPtr->FrontPtr = newNode;

		/* Inc number of elements */
		ListPtr->NumElements += 1;
	}

}

/* Add a record to the back of the list */
void AddToBackOfLinkedList(LinkedLists *ListPtr, ElementStructs *DataPtr){

	/* Variables to hold current and new node */
	LinkedListNodes *newNode;
	LinkedListNodes *curNode;

	/* Empty linked list */
	if(ListPtr->NumElements < 1){
		/* Create a new node to be added */
		newNode = (LinkedListNodes*) malloc(sizeof(LinkedListNodes));
		newNode->ElementPtr = DataPtr;

		/* Set up pointers for new node */
		newNode->Previous = NULL;
		newNode->Next = NULL;

		/* Set up pointers for list structure */
		ListPtr->FrontPtr = newNode;
		ListPtr->BackPtr = newNode;

		/* Increase list size */
		ListPtr->NumElements += 1;
	}

	/* All other list sizes */
	else{
		/* Create a new node to be added */
		newNode = (LinkedListNodes*) malloc(sizeof(LinkedListNodes));
		newNode->ElementPtr = DataPtr;

		/* Set up pointers for new node */
		curNode = ListPtr->BackPtr;

		/* Set up pointers for new node */
		newNode->Next = NULL;
		newNode->Previous = curNode;

		/* Set up pointers for current node */
		curNode->Next = newNode;

		/* Set up pointers for list sturcture */
		ListPtr->BackPtr = newNode;

		/* Increment list size */
		ListPtr->NumElements += 1;
	}
}

/* Removes and returns a record from front of list */
ElementStructs* RemoveFromFrontOfLinkedList(LinkedLists *ListPtr){
	/* Variables to hold return node, new head, and return data */
	LinkedListNodes* retNode = ListPtr->FrontPtr;
	LinkedListNodes* newHead;
	ElementStructs* retData;

	/* Empty list */
	if(retNode == NULL){
		return NULL;
	}

	/* List is now empty */
	if(retNode->Next == NULL){
		ListPtr->FrontPtr = NULL;
		ListPtr->BackPtr = NULL;
		ListPtr->NumElements -= 1;
	}

	/* All other cases */
	else{
		newHead = retNode->Next;
		newHead->Previous = NULL;
		ListPtr->FrontPtr = newHead;
		ListPtr->NumElements -= 1;
	}

	/* Free the memory allocate to node struct itself */
	retData = retNode->ElementPtr;
	free(retNode);

	return retData;
}

/* Removes and returns a record from the back of the list */
ElementStructs* RemoveFromBackOfLinkedList(LinkedLists *ListPtr){
	/* Variables to hold return node, new tail and return data */
	LinkedListNodes* retNode = ListPtr->BackPtr;
	LinkedListNodes* newTail;
	ElementStructs* retData;

	/* Empty list */
	if(retNode == NULL){
		return NULL;
	}

	/* List is now empty */
	if(retNode->Previous == NULL){
		ListPtr->BackPtr = NULL;
		ListPtr->FrontPtr = NULL;
		ListPtr->NumElements -= 1;
	}
	/* Other cases */
	else{
		newTail = retNode->Previous;
		newTail->Next = NULL;
		ListPtr->BackPtr = newTail;
		ListPtr->NumElements -= 1;
	}

	/* Free the memory allocate to node struct itself */
	retData = retNode->ElementPtr;
	free(retNode);

	return retData;
}

/* Deallocates the linked list and resets the struct fields as if the list was empty */
void DestroyLinkedList(LinkedLists *ListPtr){

	/* Variables to hold current node and next node */
	LinkedListNodes* curNode;
	LinkedListNodes* nextNode;

	/* Reset the list size */
	ListPtr->NumElements = 0;

	/* Deallocate the nodes and their contents */
	curNode = ListPtr->FrontPtr;
	nextNode = curNode->Next;
	while(curNode != NULL){
		/* Deallocate the contents */
		free(curNode->ElementPtr);
		/* Deallocate the node itself */
		free(curNode);
		/* Move onto the next node */
		curNode = nextNode;
		/* Assign a new node to the next-next node */
		if(curNode!=NULL){
			nextNode = curNode->Next;		
		}
	}
}