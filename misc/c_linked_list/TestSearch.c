/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 3,                                                      */ 
/*                                                                  */ 
/* Linked Lists - This program tests searching of the linked list   */
/* structure      													*/
/********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "LinkedLists.h"

/* Search by the actual word */
ElementStructs* SearchList(LinkedLists* theList, char* searchToken){
	LinkedListNodes* curNode = theList->FrontPtr;
	while(curNode != NULL){
		int retCode = strcmp(curNode->ElementPtr->word, searchToken);
		if(retCode == 0){
			return curNode->ElementPtr;
		}
		curNode = curNode->Next;
	}
	return NULL;
}

/* Search by an index. Index starts at 0 */
ElementStructs* SearchList_Index(LinkedLists* theList, int index){
	LinkedListNodes* curNode = theList->FrontPtr;
	int curIndex = 0;
	while(curNode != NULL){
		if(index == (curNode->ElementPtr->count)-1){
			return curNode->ElementPtr;
		}
		curNode = curNode->Next;
		curIndex += 1;
	}
	return NULL;
}

int main(int argc, char* argv[]){
	/* The linked list data set */
	LinkedLists* theList = (LinkedLists*) malloc(sizeof(LinkedLists));

	/* Data to be appended to the linked list */
	ElementStructs* theData;

	ElementStructs* searchResult;

	/* The input file */
	FILE* file;

	/* Temp variable to hold token from a line */
	char* token;

	/* Temp variable to hold a line from the file */
	char line[50];

	/* The word count and position */
	int num = 1;

	/* Check that a command line argument was provided */
	if(argc== 2){

		/* Initialize the linked list */
		InitLinkedList(theList);

		/* Open the file */
		file = fopen(argv[1], "r");

		/* Check if file has been opened successfully */
		if(file != NULL){
			/* Get a line from a file until EOF */
			while(fgets(line, sizeof line, file) != NULL){
				/* Create a new data struct */
				theData = (ElementStructs*) malloc(sizeof(ElementStructs));
				/* Remove unwanted characters from the line */
				token = strtok(line, " \t\n");
				/* Create the word token */
				strcpy((theData->word), token);
				/* Set the word position of the data */
				theData->count = num;
				/* Increment position count */
				num += 1;

				/* Add the newly created data to back of linked list */
				AddToBackOfLinkedList(theList, theData);
			}

			/* Close the file */
			fclose(file);

			/* Print the total number of words */
			printf("Total number of words: %i \n", theList->NumElements);

			/* Search and print the first 6 element */
			searchResult = SearchList(theList, "A");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "A's");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "AOL");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "AOL's");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "Aaberg");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "Aaberg's");
			printf("%s \n", searchResult->word);

			/* Search by characters and print the last last 6 elements */
			searchResult = SearchList(theList, "zymurgy");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "zymotic");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "zymosis's");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "zymosis");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "zymometer's");
			printf("%s \n", searchResult->word);
			searchResult = SearchList(theList, "zymometer");
			printf("%s \n", searchResult->word);

			/* Search by index and print the required elements */
			searchResult = SearchList_Index(theList, 0);
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, 1);
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, 2);
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, 3);
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, 4);
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, 5);
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, (theList->NumElements-1));
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, (theList->NumElements-2));
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, (theList->NumElements-3));
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, (theList->NumElements-4));
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, (theList->NumElements-5));
			printf("%s \n", searchResult->word);
			searchResult = SearchList_Index(theList, (theList->NumElements-6));
			printf("%s \n", searchResult->word);
		}
		else{
			/* The file cannot be opened */
			fprintf(stderr, "Error: File cannot be open or does not exist \n");
			return 2;
		}

		/* Destroy the linked list */
		DestroyLinkedList(theList);
	}
	else{
		/* Wrong number of arguments */
		fprintf(stderr, "Usage: %s filename \n", argv[0]);
		return 1;
	}

	/* Free the memory allocated for the linked list */
	free(theList);

	return 0;

}