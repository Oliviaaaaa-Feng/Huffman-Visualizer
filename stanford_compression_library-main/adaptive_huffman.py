from scl.compressors.vitter_adaptive_huffman_2 import VitterAdaptiveHuffmanEncoder, VitterAdaptiveHuffmanDecoder
import time

if __name__ == "__main__":
    test_file = "stanford_compression_library-main/scl/sherlock_ascii.txt"
    with open(test_file, "rb") as f:
        text = f.read()
        
    encoder = VitterAdaptiveHuffmanEncoder()
    start_time = time.time()
    bits = encoder.encode(text)
    end_time = time.time()
    encode_time = end_time - start_time

    start_time = time.time()
    decoder = VitterAdaptiveHuffmanDecoder()

    recovered = decoder.decode(bits)
    end_time = time.time()
    decode_time = end_time - start_time
    
    print("Recovered:", recovered.decode('utf-8'))
    print("Bits length:", len(bits))
    print("Match?", recovered == text)
    print(f"Encoding time: {encode_time} seconds")
    print(f"Decoding time: {decode_time} seconds")