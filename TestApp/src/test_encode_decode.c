// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "slapcodec.h"
#include <time.h>

int main(int argc, char **argv)
{
  void *pBuffer = NULL;
  void *pBuffer2 = NULL;
  int retval = 0;

  if (argc > 2)
  {
    FILE *pFile = fopen(argv[1], "rb");

    if (!pFile)
    {
      printf("File not found.");
      retval = 1;
      goto epilogue;
    }

    fseek(pFile, 0, SEEK_END);
    size_t size = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);

    pBuffer = malloc(7680 * 11520);
    printf("Start: 0x%" PRIx64 ", End: 0x%" PRIx64 ".\n", (uint64_t)pBuffer, (uint64_t)((uint8_t *)pBuffer + 7680 * 11520));

    if (!pBuffer)
    {
      printf("Memory allocation failure.");
      retval = 1;
      goto epilogue;
    }

    size_t readBytes = fread(pBuffer, 1, size, pFile);
    printf("Read %" PRIu64 " bytes from '%s'.\n", readBytes, argv[1]);

    fclose(pFile);
  }
  else
  {
    printf("Usage: %s <inputfile> <outputfile>", argv[0]);
    goto epilogue;
  }

  printf("Creating File Writer...\n");
  slapFileWriter *pFileWriter = slapCreateFileWriter(argv[2], 7680, 7680, SLAP_FLAG_STEREO);

  size_t frameCount = 10;
  printf("Adding %" PRIu64 " frames...\n", frameCount);

  pBuffer2 = malloc(7680 * 11520);
  clock_t before = clock();

  for (size_t i = 0; i < frameCount; i++)
  {
    slapMemcpy(pBuffer2, pBuffer, 7680 * 11520);
    slapFileWriter_AddFrameYUV420(pFileWriter, pBuffer2);
  }

  clock_t after = clock();

  printf("%d ms -> ~%f ms / frame\n", after - before, (after - before) / (float)frameCount);

  printf("Finalizing File...\n");
  slapFinalizeFileWriter(pFileWriter);

  printf("Destroying File Writer...\n");
  slapDestroyFileWriter(&pFileWriter);

  printf("Encoding Done.\n");

  printf("Creating File Reader...\n");
  slapFileReader *pFileReader = slapCreateFileReader(argv[2]);

  slapResult result;
  frameCount = 0;

  printf("Decoding Frames...\n");

  before = clock();

#define SAVE_AS_JPEG 1

  do
  {
    result = _slapFileReader_ReadNextFrame(pFileReader);

    if (result != slapSuccess)
      break;

    result = _slapFileReader_DecodeCurrentFrameFull(pFileReader);

    if (result != slapSuccess)
      break;

    frameCount++;

#ifdef SAVE_AS_JPEG
    char fname[255];
    sprintf_s(fname, 255, "%s-%" PRIu64 ".jpg", argv[2], frameCount);

    slapWriteJpegFromYUV(fname, pFileReader->pDecodedFrameYUV, 7680, 7680);
#endif
  } while (1);
  
  after = clock();

#ifndef SAVE_AS_JPEG
  printf("%d ms -> ~%f ms / frame\n", after - before, (after - before) / (float)frameCount);
#endif

  printf("Frame Count: %" PRIu64 ".\n", frameCount);

  printf("Destroying File Reader...\n");
  slapDestroyFileReader(&pFileReader);

epilogue:
  free(pBuffer);
  return retval;
}
