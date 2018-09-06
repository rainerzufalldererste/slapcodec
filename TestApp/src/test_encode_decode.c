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

    pBuffer = malloc(size);

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

  clock_t before = clock();

  for (size_t i = 0; i < frameCount; i++)
    slapFileWriter_AddFrameYUV420(pFileWriter, pBuffer);

  clock_t after = clock();

  printf("%d ms.\n", after - before);

  printf("Finalizing File...\n");
  slapFinalizeFileWriter(pFileWriter);

  printf("Destroying File Writer...\n");
  slapDestroyFileWriter(&pFileWriter);

  printf("Encoding Done.\n");

  printf("Creating File Reader...\n");
  slapFileReader *pFileReader = slapCreateFileReader(argv[2]);

  slapResult result;
  frameCount = 0;
  
  do
  {
    result = _slapFileReader_ReadNextFrame(pFileReader);

    if (result != slapSuccess)
      break;

    frameCount++;

    char fname[255];
    sprintf_s(fname, 255, "%s-%" PRIu64 ".jpg", argv[2], frameCount);

    FILE *pFrame = fopen(fname, "wb");

    fwrite(pFileReader->pCurrentFrame, 1, pFileReader->currentFrameSize, pFrame);
    fclose(pFrame);

  } while (1);

  printf("Frame Count: %" PRIu64 ".\n", frameCount);

  printf("Destroying File Reader...\n");
  slapDestroyFileReader(&pFileReader);

epilogue:
  free(pBuffer);
  return retval;
}
