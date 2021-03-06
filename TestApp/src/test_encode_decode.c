// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "slapcodec.h"
#include <time.h>

#define ASSERT_SUCCESS(function) do { if ((function) != slapSuccess) __debugbreak(); } while (0)

int main(int argc, char **argv)
{
  void *pFileData = NULL;
  void *pFrame = NULL;
  int retval = 0;
  char *origFile = NULL;
  char *slapFile = NULL;

  if (argc > 2)
  {
    origFile = argv[1];
    slapFile = argv[2];

    FILE *pFile = fopen(origFile, "rb");

    if (!pFile)
    {
      printf("File not found.");
      retval = 1;
      goto epilogue;
    }

    fseek(pFile, 0, SEEK_END);
    size_t size = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);

    pFileData = malloc(7680 * 11520);

    if (!pFileData)
    {
      printf("Memory allocation failure.");
      retval = 1;
      goto epilogue;
    }

    size_t readBytes = fread(pFileData, 1, size, pFile);
    printf("Read %" PRIu64 " bytes from '%s'.\n", readBytes, argv[1]);

    fclose(pFile);
  }
  else if (argc == 2)
  {
    slapFile = argv[1];
  }
  else
  {
    printf("Usage: %s <inputfile> <outputfile>", argv[0]);
    goto epilogue;
  }

  size_t frameCount;
  clock_t before;
  clock_t after;

  if (origFile)
  {
    printf("Creating File Writer...\n");
    slapFileWriter *pFileWriter = slapCreateFileWriter(slapFile, 7680, 7680, SLAP_FLAG_STEREO);

    frameCount = 100;
    printf("Adding %" PRIu64 " frames...\n", frameCount);

    pFrame = malloc(7680 * 11520);
    before = clock();

    for (size_t i = 0; i < frameCount; i++)
    {
      slapMemcpy(pFrame, pFileData, 7680 * 11520);
      ASSERT_SUCCESS(slapFileWriter_AddFrameYUV420(pFileWriter, pFrame));
      printf("\rFrame %" PRIu64 " / %" PRIu64 " processed.", i + 1, frameCount);
    }

    printf("\r");

    after = clock();

    printf("%d ms -> ~%f ms / frame\n", after - before, (after - before) / (float)frameCount);

    printf("Finalizing File...\n");
    ASSERT_SUCCESS(slapFinalizeFileWriter(pFileWriter));

    printf("Destroying File Writer...\n");
    slapDestroyFileWriter(&pFileWriter);

    printf("Encoding Done.\n");
  }

  printf("Creating File Reader...\n");
  slapFileReader *pFileReader = slapCreateFileReader(slapFile);

  slapResult result;
  frameCount = 0;

  printf("Decoding Frames...\n");

  before = clock();

//#define SAVE_AS_JPEG 1
//#define DECODE_LOW_RES 1

#ifdef SAVE_AS_JPEG
//#define SAVE_INTERNAL_FRAMES
#endif

  do
  {
#if !DECODE_LOW_RES
    result = _slapFileReader_ReadNextFrameFull(pFileReader);

    if (result != slapSuccess)
      break;

    result = _slapFileReader_DecodeCurrentFrameFull(pFileReader);

    if (result != slapSuccess)
      break;

#ifdef SAVE_INTERNAL_FRAMES
    char fname0[255];
    sprintf_s(fname0, 255, "%s-%" PRIu64 ".raw.jpg", slapFile, frameCount);

    FILE *pRAW = fopen(fname0, "wb");
    fwrite(pFileReader->pCurrentFrame, 1, pFileReader->currentFrameSize, pRAW);
    fclose(pRAW);
#endif

#else
    result = _slapFileReader_ReadNextFrameLowRes(pFileReader);

    if (result != slapSuccess)
      break;

    result = _slapFileReader_DecodeCurrentFrameLowRes(pFileReader);

    if (result != slapSuccess)
      break;
#endif

    frameCount++;

#ifdef SAVE_AS_JPEG
    char fname[255];
    sprintf_s(fname, 255, "%s-%" PRIu64 ".jpg", slapFile, frameCount);
    size_t resX, resY;

#if !DECODE_LOW_RES
    slapFileReader_GetResolution(pFileReader, &resX, &resY);
    slapWriteJpegFromYUV(fname, pFileReader->pDecodedFrameYUV, 7680, 7680);
#else
    slapFileReader_GetLowResFrameResolution(pFileReader, &resX, &resY);
    slapWriteJpegFromYUV(fname, pFileReader->pDecodedFrameYUV, resX, resY);
#endif
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
  free(pFileData);
  free(pFrame);
  return retval;
}
