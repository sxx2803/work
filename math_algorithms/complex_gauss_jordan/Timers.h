#ifndef _TIMERS_H_
#define _TIMERS_H_
#define NUM_ITERATIONS 1

#if defined(EN_TIME)
	#include <stdio.h>
	#include <time.h>
	#if defined(WARNING_MSG)
		#warning Timers enabled! Execution could be adversely affected.
	#endif
#endif

#if defined(EN_TIME)
	#define DECLARE_TIMER(A)				\
		struct{						\
    		clock_t Start;					\
    		clock_t Stop;					\
    		clock_t Elapsed;				\
    		int State;					\
		} 						\
		A = {						\
			0,					\
			0,					\
			0,					\
			0,					\
		};

	#define START_TIMER(A)					\
		{						\
			if(1 == A.State)			\
				fprintf(stderr, "Error, running timer "#A" started.\n");    \
			A.State = 1;				\
			A.Start = clock();			\
		}

	#define RESET_TIMER(A)					\
		{						\
			A.Elapsed = 0;				\
		}						\

	#define STOP_TIMER(A)					\
		{						\
			A.Stop = clock();			\
			if(0 == A.State)			\
				fprintf(stderr, "Error, stopped timer "#A" stopped again.\n"); \
			else 					\
				A.Elapsed += A.Stop - A.Start;	\
			A.State = 0;				\
		}

	#define PRINT_TIMER(A)					\
		{						\
			if(1 == A.State)			\
				STOP_TIMER(A)			\
			fprintf(stderr, "Time per calculation = %5.3f msec.\n",		\
								((double)A.Elapsed / (double)CLOCKS_PER_SEC) / (double)NUM_ITERATIONS  * 1000.0);\
		}

#else

	#define DECLARE_TIMER(A)
	#define START_TIMER(A)
	#define RESET_TIMER(A)
	#define STOP_TIMER(A)
	#define PRINT_TIMER(A)

#endif

#endif
