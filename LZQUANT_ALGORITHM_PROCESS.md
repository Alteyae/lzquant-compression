# LZQuant Compression Algorithm Process

## Algorithm Core Process

LZQuant combines color quantization and dictionary-based compression through a multi-stage process. Here is the detailed algorithm for each step:

### 1. Image Preparation Phase

```
function prepareImage(inputImage):
    // Convert to suitable format for processing
    if inputImage is not PNG:
        convert inputImage to PNG format
    
    // Save original for quality comparison later
    originalImage = deepCopy(inputImage)
    originalSize = size(inputImage)
    
    return {originalImage, originalSize, pngImage}
```

### 2. Color Quantization Phase

```
function applyColorQuantization(pngImage, quality):
    // Initialize PNGQuant parameters
    minQuality = max(0, quality - 10)
    maxQuality = quality
    qualityRange = minQuality + "-" + maxQuality
    
    // Execute PNGQuant with parameters
    result = execute("pngquant", [
        "--force",
        "--output", temporaryOutputPath,
        "--quality", qualityRange,
        inputPath
    ])
    
    // Error handling
    if result.failed:
        throw new Error("PNGQuant compression failed")
    
    // Measure results
    quantizedSize = fileSize(temporaryOutputPath)
    quantizedImage = readFile(temporaryOutputPath)
    
    // Calculate intermediate compression ratio
    quantizationRatio = ((originalSize - quantizedSize) / originalSize) * 100
    
    return {quantizedImage, quantizedSize, quantizationRatio}
```

### 3. LZ4 Compression Phase

```
function applyLZ4Compression(quantizedImage, metadata):
    // Apply LZ4 Frame compression
    compressedData = lz4.compress(
        quantizedImage,
        compressionLevel = LZ4_MAX_COMPRESSION_LEVEL
    )
    
    // Create metadata header
    metadataJSON = convertToJSON(metadata)
    headerSize = length(metadataJSON)
    
    // Assemble PLZ file structure
    plzFile = concatenate([
        PLZ_MAGIC_NUMBER,              // 4 bytes: "PLZ\0"
        intToBytes(headerSize, 4),     // 4 bytes: header size as uint32 little-endian
        metadataJSON,                  // variable length: JSON metadata
        compressedData                 // remaining: LZ4 compressed image
    ])
    
    // Calculate compression metrics
    finalSize = length(plzFile)
    lz4Ratio = ((quantizedSize - finalSize) / quantizedSize) * 100
    totalRatio = ((originalSize - finalSize) / originalSize) * 100
    
    return {plzFile, finalSize, lz4Ratio, totalRatio}
```

### 4. Decompression Phase

```
function decompressLZQuant(plzFile):
    // Read and validate file structure
    if plzFile[0:4] != PLZ_MAGIC_NUMBER:
        throw new Error("Invalid PLZ file format")
    
    // Extract header
    headerSize = bytesToInt(plzFile[4:8])
    headerJSON = plzFile[8:(8+headerSize)]
    metadata = parseJSON(headerJSON)
    
    // Extract and decompress data
    compressedData = plzFile[(8+headerSize):]
    decompressedPNG = lz4.decompress(compressedData)
    
    return {decompressedPNG, metadata}
```

### 5. Quality Assessment Phase

```
function assessQuality(originalImage, decompressedImage):
    // Convert both to same color space for comparison
    originalRGB = convertToRGB(originalImage)
    decompressedRGB = convertToRGB(decompressedImage)
    
    // Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = calculateMeanSquaredError(originalRGB, decompressedRGB)
    psnr = 10 * log10((MAX_PIXEL_VALUE^2) / mse)
    
    // Calculate SSIM (Structural Similarity Index)
    ssim = calculateSSIM(originalRGB, decompressedRGB)
    
    return {psnr, ssim}
```

## Complete LZQuant Algorithm

Here is the complete algorithm combining all phases:

```
function compressWithLZQuant(inputImage, quality = 80, customMetadata = {}):
    // Record start time and performance metrics
    startTime = currentTime()
    systemResources = monitorSystemResources()
    
    try:
        // Phase 1: Image Preparation
        {originalImage, originalSize, pngImage} = prepareImage(inputImage)
        
        // Phase 2: Color Quantization
        {quantizedImage, quantizedSize, quantizationRatio} = applyColorQuantization(pngImage, quality)
        
        // Record compression time
        compressionTime = currentTime() - startTime
        
        // Prepare metadata
        metadata = {
            compressionQuality: quality,
            createdTimestamp: startTime,
            version: "1.0",
            ...customMetadata
        }
        
        // Phase 3: LZ4 Compression
        {plzFile, finalSize, lz4Ratio, totalRatio} = applyLZ4Compression(quantizedImage, metadata)
        
        // Phase 5: Decompression for quality assessment
        decompStartTime = currentTime()
        {decompressedPNG, _} = decompressLZQuant(plzFile)
        decompressionTime = currentTime() - decompStartTime
        
        // Phase 5: Quality Assessment
        {psnr, ssim} = assessQuality(originalImage, loadImage(decompressedPNG))
        
        // Final system resource check
        finalResources = monitorSystemResources()
        
        // Compile results
        compressionResults = {
            originalSize: originalSize,
            quantizedSize: quantizedSize,
            finalSize: finalSize,
            quantizationRatio: quantizationRatio,
            lz4Ratio: lz4Ratio,
            totalRatio: totalRatio,
            compressionTime: compressionTime,
            decompressionTime: decompressionTime,
            psnr: psnr,
            ssim: ssim,
            cpuUsage: finalResources.cpuUsage,
            memoryUsage: finalResources.memoryUsage
        }
        
        return {plzFile, compressionResults}
        
    } catch (error) {
        // Clean up any temporary files
        cleanupTemporaryFiles()
        throw error
    }
```

## Key Algorithmic Features

### Color Quantization Logic

The quantization algorithm in PNGQuant:

1. **Palette Generation**:
   ```
   function generateOptimalPalette(image, maxColors):
       // Initialize color space
       colorSpace = createColorCube(image)
       
       // Iteratively split the color cube in the direction of maximum variance
       while numberOfColors < maxColors:
           // Find box with greatest color variance
           boxToSplit = findBoxWithGreatestVariance(colorBoxes)
           
           // Split the box at median point along largest variance axis
           {box1, box2} = splitBox(boxToSplit)
           
           // Replace the original box with the two new boxes
           colorBoxes.remove(boxToSplit)
           colorBoxes.add(box1)
           colorBoxes.add(box2)
       
       // Calculate representative color for each box (weighted average)
       for each box in colorBoxes:
           palette.add(calculateRepresentativeColor(box))
       
       return palette
   ```

2. **Dithering Algorithm**:
   ```
   function applyFloydSteinbergDithering(image, palette):
       // For each pixel in the image (from top to bottom, left to right)
       for y = 0 to height-1:
           for x = 0 to width-1:
               oldPixel = image[x][y]
               newPixel = findClosestPaletteColor(oldPixel, palette)
               image[x][y] = newPixel
               
               // Calculate quantization error
               error = oldPixel - newPixel
               
               // Distribute error to neighboring pixels
               if x+1 < width:
                   image[x+1][y] += error * 7/16
               if x-1 >= 0 and y+1 < height:
                   image[x-1][y+1] += error * 3/16
               if y+1 < height:
                   image[x][y+1] += error * 5/16
               if x+1 < width and y+1 < height:
                   image[x+1][y+1] += error * 1/16
       
       return image
   ```

### LZ4 Compression Logic

The core LZ4 compression algorithm operates as follows:

```
function lz4Compress(data):
    // Initialize compression structures
    hashTable = initializeHashTable()
    output = []
    
    position = 0
    while position < length(data):
        // Look for a match in previously seen data
        matchLength = 0
        matchOffset = 0
        
        // Hash current 4 bytes to find potential match positions
        hash = hashFunction(data[position:position+4])
        matchPosition = hashTable[hash]
        
        if matchPosition exists and position - matchPosition < MAX_DISTANCE:
            // Calculate match length
            while data[position + matchLength] == data[matchPosition + matchLength]:
                matchLength++
                if position + matchLength >= length(data):
                    break
        
        // Update hash table with current position
        hashTable[hash] = position
        
        if matchLength >= MIN_MATCH_LENGTH:
            // Encode a match (offset and length)
            output.append(encodeMatch(position - matchPosition, matchLength))
            position += matchLength
        else:
            // Encode a literal (uncompressed byte)
            output.append(encodeLiteral(data[position]))
            position++
    
    return output
```

## Optimizations

The LZQuant algorithm includes several optimizations:

1. **Quality Parameter Tuning**:
   ```
   function calculateOptimalQuality(targetRatio, imageComplexity):
       // Start with default quality
       quality = 80
       
       // Adjust based on image complexity
       if imageComplexity == "high":  // High frequency details
           quality = min(quality + 10, 100)
       else if imageComplexity == "low":  // Low frequency, large color areas
           quality = max(quality - 10, 30)
       
       // Adjust based on target ratio
       if targetRatio > 80:  // Need very high compression
           quality = max(quality - 15, 20)
       else if targetRatio < 40:  // Need minimal compression
           quality = min(quality + 15, 95)
       
       return quality
   ```

2. **LZ4 Frame Configuration**:
   ```
   function configureLZ4Parameters(dataSize):
       if dataSize > 10 * 1024 * 1024:  // > 10MB
           // For large files, focus on compression ratio
           return {
               compressionLevel: LZ4_MAX_COMPRESSION_LEVEL,
               blockSize: 4 * 1024 * 1024,  // 4MB blocks
               contentChecksumFlag: true     // Add content checksum
           }
       else:
           // For smaller files, default parameters work well
           return {
               compressionLevel: LZ4_MAX_COMPRESSION_LEVEL,
               blockSize: 64 * 1024,  // 64KB blocks
               contentChecksumFlag: false
           }
   }
   ```

## Complexity Analysis

- **Time Complexity**:
  - Color Quantization: O(n * log(k)) where n is pixel count and k is color count
  - LZ4 Compression: O(n) where n is data size
  - Overall: O(n * log(k)) dominated by quantization

- **Space Complexity**:
  - Temporary storage: O(n) for original and intermediate images
  - Hash table for LZ4: O(n) in worst case
  - Overall: O(n)

## Error Handling and Edge Cases

```
function handleEdgeCases(image):
    // Check for extremely small images
    if width(image) < 16 or height(image) < 16:
        // For tiny images, skip quantization and just use lossless compression
        return {skipQuantization: true}
    
    // Check for images with very few colors already
    uniqueColors = countUniqueColors(image)
    if uniqueColors < 32:
        // For images with few colors, use lossless compression with minimal quantization
        return {minQuality: 90, maxQuality: 100}
    
    // Handle grayscale images differently
    if isGrayscale(image):
        // For grayscale, optimize palette differently
        return {grayscaleOptimized: true}
        
    return {skipQuantization: false}
```

This detailed algorithm description provides a comprehensive view of how LZQuant processes images through every step from input to compressed output.