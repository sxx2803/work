/********************************************************************/ 
/* Sicheng Xu (Fall Quarter)                                        */ 
/* sxx2803@rit.edu  					           				    */ 
/* Homework 3,                                                      */ 
/*                                                                  */ 
/* Linked Lists - This program tests the linked list structure      */
/* itself														    */
/********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "LinkedLists.h"

int main(int argc, char* argv[]){
	/* The linked list data set */
	LinkedLists* theList = (LinkedLists*) malloc(sizeof(LinkedLists));

	/* Data to be appended to the linked list */
	ElementStructs* theData;

	/* The input file */
	FILE* file;

	/* Temp variable to hold token from a line */
	char* token;

	/* Temp variable to hold a line from the file */
	char line[50];

	/* The word count and position */
	int num = 1;

	/* Temp variable to hold data returned from linked list */
	ElementStructs* tempVar;

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

			/* Print the first 6 element and free each elementstruct */
			tempVar = RemoveFromFrontOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromFrontOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromFrontOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromFrontOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromFrontOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromFrontOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);

			/* Print the last last 6 elements and free each elementstruct*/
			tempVar = RemoveFromBackOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromBackOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromBackOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromBackOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromBackOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
			tempVar = RemoveFromBackOfLinkedList(theList);
			printf("%s \n", tempVar->word);
			free(tempVar);
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