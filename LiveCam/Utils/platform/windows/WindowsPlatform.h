#ifndef LIVECAM_PLATFORM_WINDOWSH
#define LIVECAM_PLATFORM_WINDOWSH

// See libebml, libbml_t.h for platform specific types


//#define WINVER 0x0502
//#define _WIN32_WINNT 0x0502
//#include <time.h>
//#include <windows.h>
/*
#if (_MSC_VER < 1300)
   typedef signed char       int8_t;
   typedef signed short      int16_t;
   #ifndef int32_t
 	  typedef signed int        int32_t;
   #endif
   
   typedef unsigned char     uint8_t;
   typedef unsigned short    uint16_t;
   typedef unsigned int      uint32_t;
#else
   typedef signed __int8     int8_t;
   typedef signed __int16    int16_t;
   #ifndef int32_t
//  	 typedef int    int32_t;
   #endif
   typedef unsigned __int8   uint8_t;
   typedef unsigned __int16  uint16_t;
   typedef unsigned __int32  uint32_t;
#endif

typedef signed __int64       int64_t;
typedef unsigned __int64     uint64_t;
*/

#include <time.h>
#include <windows.h>

#if (_MSC_VER < 1300)
   typedef signed char       lc_int8;
   typedef signed short      lc_int16;
   typedef signed char       lc_int8;
   typedef signed int        lc_int32;
   
   typedef unsigned char     lc_uint8;
   typedef unsigned short    lc_uint16;
   typedef unsigned int      lc_uint32;
#else
   typedef signed __int8     lc_int8;
   typedef signed __int16    lc_int16;
   typedef signed __int32    lc_int32;
   typedef unsigned __int8   lc_uint8;
   typedef unsigned __int16  lc_uint16;
   typedef unsigned __int32  lc_uint32;
#endif

typedef signed __int64       lc_int64;
typedef unsigned __int64     lc_uint64;

#define atoll atol
	
#endif
