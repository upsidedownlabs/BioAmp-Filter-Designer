#!/usr/bin/env python3

usage_guide = """\
This script generates complete filter implementations (not just coefficients) for
digital IIR filters using the SciPy signal processing library. It uses the
well-known Butterworth design which optimizes flat frequency response.

The script generates ready-to-use filter classes/functions in multiple languages:
- Python: Class with process() and reset() methods
- C++: Class with process() and reset() methods (like your example)
- JavaScript: Class with process() and reset() methods
- TypeScript: Class with process() and reset() methods (with types)
- Java: Class with process() and reset() methods

The filter type determines the range of frequencies to block. Lowpass smooths
signals by blocking high frequencies; highpass removes constant components by
blocking low frequencies; bandpass blocks both low and high frequencies to
emphasize a range of interest; bandstop blocks a range of frequencies to remove
specific unwanted frequency components such as periodic noise sources.

The sampling frequency is the constant rate at which the sensor signal is
sampled and is specified in samples per second.

The filter order determines both the number of state variables and steepness of
frequency response. It specifies the number of terms in the frequency-space
polynomials which define the filter.

For lowpass and highpass filters, the critical frequency specifies the corner of
the idealized filter curve in Hz. The filter has a rolloff, so the blocking
strength increases as frequencies move beyond this corner into the blocked range.

For bandpass and bandstop filters, the critical frequency is the center of the
band, and the width is the total width of the band.

References:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html
https://en.wikipedia.org/wiki/Butterworth_filter
https://en.wikipedia.org/wiki/Digital_filter#Direct_form_II
"""

################################################################
# Standard Python libraries.
import sys, argparse, logging, os

# Set up debugging output.
# logging.getLogger().setLevel(logging.DEBUG)

# Extension libraries.
import numpy as np
import scipy.signal

type_print_form = {'lowpass': 'Low-Pass', 'highpass': 'High-Pass', 'bandpass': 'Band-Pass', 'bandstop': 'Band-Stop'}

################################################################
# Optionally generate plots of the filter properties.
def make_plots(filename, sos, fs, order, freqs, name):
    try:
        import matplotlib.pyplot as plt
    except:
        print("Warning, matplotlib not found, skipping plot generation.")
        return

    # N.B. response is a vector of complex numbers
    freq, response = scipy.signal.sosfreqz(sos, fs=fs)

    fig, ax = plt.subplots(nrows=1)
    fig.set_dpi(160)
    fig.set_size_inches((8,6))

    ax.plot(freq, np.abs(response))
    ax.set_title(f"Response of {freqs} Hz {name} Filter of Order {order}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude of Transfer Ratio")

    fig.savefig(filename)

################################################################
def emit_python_filter(stream, name, sos, filter_type, fs, freqs, order):
    """Generate Python filter class"""
    sections = len(sos)

    # Header comment
    stream.write(f"# {type_print_form[filter_type]} Butterworth IIR digital filter\n")
    stream.write(f"# Sampling rate: {fs} Hz, frequency: {freqs} Hz\n")
    stream.write(f"# Filter is order {order}, implemented as second-order sections (biquads)\n")
    stream.write("# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html\n\n")

    stream.write(f"class {name}:\n")
    stream.write("    def __init__(self):\n")
    stream.write("        # Initialize state variables for each biquad section\n")

    for i in range(sections):
        stream.write(f"        self.z1_{i} = 0.0\n")
        stream.write(f"        self.z2_{i} = 0.0\n")

    stream.write("\n")
    stream.write("    def process(self, input_sample):\n")
    stream.write("        \"\"\"Process a single sample through the filter\"\"\"\n")
    stream.write("        output = input_sample\n")

    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        stream.write(f"\n        # Biquad section {i}\n")
        stream.write(f"        x = output - ({a1:.8f} * self.z1_{i}) - ({a2:.8f} * self.z2_{i})\n")
        stream.write(f"        output = {b0:.8f} * x + {b1:.8f} * self.z1_{i} + {b2:.8f} * self.z2_{i}\n")
        stream.write(f"        self.z2_{i} = self.z1_{i}\n")
        stream.write(f"        self.z1_{i} = x\n")

    stream.write("\n        return output\n\n")

    stream.write("    def reset(self):\n")
    stream.write("        \"\"\"Reset filter state variables\"\"\"\n")
    for i in range(sections):
        stream.write(f"        self.z1_{i} = 0.0\n")
        stream.write(f"        self.z2_{i} = 0.0\n")

    # Add simple example usage
    stream.write("\n\n# Example usage:\n")
    stream.write("# Single channel:\n")
    stream.write(f"# filter = {name}()\n")
    stream.write("# filter.reset()\n")
    stream.write("# filtered_output = filter.process(sample)\n")
    stream.write("# \n")
    stream.write("# Multi-channel (3 channels):\n")
    stream.write(f"# filters = [{name}() for _ in range(3)]  # One filter per channel\n")
    stream.write("# filtered_1 = filters[0].process(raw1)\n")
    stream.write("# filtered_2 = filters[1].process(raw2)\n")
    stream.write("# filtered_3 = filters[2].process(raw3)\n")

################################################################
def emit_cpp_filter(stream, name, sos, filter_type, fs, freqs, order):
    """Generate C++ filter class"""
    sections = len(sos)

    # Header comment
    stream.write(f"// {type_print_form[filter_type]} Butterworth IIR digital filter\n")
    stream.write(f"// Sampling rate: {fs} Hz, frequency: {freqs} Hz\n")
    stream.write(f"// Filter is order {order}, implemented as second-order sections (biquads)\n")
    stream.write("// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html\n\n")

    stream.write("#include <iostream>\n\n")

    stream.write(f"class {name} {{\n")
    stream.write("private:\n")
    stream.write("    struct BiquadState { float z1 = 0, z2 = 0; };\n")

    for i in range(sections):
        stream.write(f"    BiquadState state{i};\n")

    stream.write("\npublic:\n")
    stream.write("    float process(float input) {\n")
    stream.write("        float output = input;\n")

    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        stream.write(f"\n        // Biquad section {i}\n")
        stream.write(f"        float x{i} = output - ({a1:.8f}f * state{i}.z1) - ({a2:.8f}f * state{i}.z2);\n")
        stream.write(f"        output = {b0:.8f}f * x{i} + {b1:.8f}f * state{i}.z1 + {b2:.8f}f * state{i}.z2;\n")
        stream.write(f"        state{i}.z2 = state{i}.z1;\n")
        stream.write(f"        state{i}.z1 = x{i};\n")

    stream.write("\n        return output;\n")
    stream.write("    }\n\n")

    stream.write("    void reset() {\n")
    for i in range(sections):
        stream.write(f"        state{i}.z1 = state{i}.z2 = 0;\n")
    stream.write("    }\n")
    stream.write("};\n\n")

    # Add simple example usage
    stream.write("// Example usage:\n")
    stream.write("// Single channel:\n")
    stream.write(f"// {name} filter;\n")
    stream.write("// filter.reset();\n")
    stream.write("// float filtered_output = filter.process(sample);\n")
    stream.write("// \n")
    stream.write("// Multi-channel (3 channels):\n")
    stream.write(f"// {name} filters[3];  // One filter per channel\n")
    stream.write("// float filtered_1 = filters[0].process(raw1);\n")
    stream.write("// float filtered_2 = filters[1].process(raw2);\n")
    stream.write("// float filtered_3 = filters[2].process(raw3);\n")

################################################################
def emit_javascript_filter(stream, name, sos, filter_type, fs, freqs, order):
    """Generate JavaScript filter class"""
    sections = len(sos)

    # Header comment
    stream.write(f"// {type_print_form[filter_type]} Butterworth IIR digital filter\n")
    stream.write(f"// Sampling rate: {fs} Hz, frequency: {freqs} Hz\n")
    stream.write(f"// Filter is order {order}, implemented as second-order sections (biquads)\n")
    stream.write("// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html\n\n")

    stream.write(f"class {name} {{\n")
    stream.write("    constructor() {\n")
    stream.write("        // Initialize state variables for each biquad section\n")

    for i in range(sections):
        stream.write(f"        this.z1_{i} = 0.0;\n")
        stream.write(f"        this.z2_{i} = 0.0;\n")

    stream.write("    }\n\n")

    stream.write("    process(inputSample) {\n")
    stream.write("        let output = inputSample;\n")

    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        stream.write(f"\n        // Biquad section {i}\n")
        stream.write(f"        let x{i} = output - ({a1:.8f} * this.z1_{i}) - ({a2:.8f} * this.z2_{i});\n")
        stream.write(f"        output = {b0:.8f} * x{i} + {b1:.8f} * this.z1_{i} + {b2:.8f} * this.z2_{i};\n")
        stream.write(f"        this.z2_{i} = this.z1_{i};\n")
        stream.write(f"        this.z1_{i} = x{i};\n")

    stream.write("\n        return output;\n")
    stream.write("    }\n\n")

    stream.write("    reset() {\n")
    for i in range(sections):
        stream.write(f"        this.z1_{i} = 0.0;\n")
        stream.write(f"        this.z2_{i} = 0.0;\n")
    stream.write("    }\n")
    stream.write("}\n\n")

    # Add simple example usage
    stream.write("// Example usage:\n")
    stream.write("// Single channel:\n")
    stream.write(f"// const filter = new {name}();\n")
    stream.write("// filter.reset();\n")
    stream.write("// const filtered_output = filter.process(sample);\n")
    stream.write("// \n")
    stream.write("// Multi-channel (3 channels):\n")
    stream.write(f"// const filters = Array(3).fill().map(() => new {name}());  // One filter per channel\n")
    stream.write("// const filtered_1 = filters[0].process(raw1);\n")
    stream.write("// const filtered_2 = filters[1].process(raw2);\n")
    stream.write("// const filtered_3 = filters[2].process(raw3);\n")

################################################################
def emit_typescript_filter(stream, name, sos, filter_type, fs, freqs, order):
    """Generate TypeScript filter class"""
    sections = len(sos)

    # Header comment
    stream.write(f"// {type_print_form[filter_type]} Butterworth IIR digital filter\n")
    stream.write(f"// Sampling rate: {fs} Hz, frequency: {freqs} Hz\n")
    stream.write(f"// Filter is order {order}, implemented as second-order sections (biquads)\n")
    stream.write("// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html\n\n")

    stream.write(f"class {name} {{\n")

    for i in range(sections):
        stream.write(f"    private z1_{i}: number = 0.0;\n")
        stream.write(f"    private z2_{i}: number = 0.0;\n")

    stream.write("\n")

    stream.write("    process(inputSample: number): number {\n")
    stream.write("        let output: number = inputSample;\n")

    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        stream.write(f"\n        // Biquad section {i}\n")
        stream.write(f"        const x{i}: number = output - ({a1:.8f} * this.z1_{i}) - ({a2:.8f} * this.z2_{i});\n")
        stream.write(f"        output = {b0:.8f} * x{i} + {b1:.8f} * this.z1_{i} + {b2:.8f} * this.z2_{i};\n")
        stream.write(f"        this.z2_{i} = this.z1_{i};\n")
        stream.write(f"        this.z1_{i} = x{i};\n")

    stream.write("\n        return output;\n")
    stream.write("    }\n\n")

    stream.write("    reset(): void {\n")
    for i in range(sections):
        stream.write(f"        this.z1_{i} = 0.0;\n")
        stream.write(f"        this.z2_{i} = 0.0;\n")
    stream.write("    }\n")
    stream.write("}\n\n")

    # Add simple example usage
    stream.write("// Example usage:\n")
    stream.write("// Single channel:\n")
    stream.write(f"// const filter: {name} = new {name}();\n")
    stream.write("// filter.reset();\n")
    stream.write("// const filtered_output: number = filter.process(sample);\n")
    stream.write("// \n")
    stream.write("// Multi-channel (3 channels):\n")
    stream.write(f"// const filters: {name}[] = Array(3).fill(null).map(() => new {name}());  // One filter per channel\n")
    stream.write("// const filtered_1: number = filters[0].process(raw1);\n")
    stream.write("// const filtered_2: number = filters[1].process(raw2);\n")
    stream.write("// const filtered_3: number = filters[2].process(raw3);\n")

################################################################
def emit_java_filter(stream, name, sos, filter_type, fs, freqs, order):
    """Generate Java filter class"""
    sections = len(sos)

    # Header comment
    stream.write(f"// {type_print_form[filter_type]} Butterworth IIR digital filter\n")
    stream.write(f"// Sampling rate: {fs} Hz, frequency: {freqs} Hz\n")
    stream.write(f"// Filter is order {order}, implemented as second-order sections (biquads)\n")
    stream.write("// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html\n\n")

    stream.write(f"public class {name} {{\n")

    for i in range(sections):
        stream.write(f"    private double z1_{i} = 0.0;\n")
        stream.write(f"    private double z2_{i} = 0.0;\n")

    stream.write("\n")

    stream.write("    public double process(double inputSample) {\n")
    stream.write("        double output = inputSample;\n")

    for i, section in enumerate(sos):
        b0, b1, b2, a0, a1, a2 = section
        stream.write(f"\n        // Biquad section {i}\n")
        stream.write(f"        double x{i} = output - ({a1:.8f} * this.z1_{i}) - ({a2:.8f} * this.z2_{i});\n")
        stream.write(f"        output = {b0:.8f} * x{i} + {b1:.8f} * this.z1_{i} + {b2:.8f} * this.z2_{i};\n")
        stream.write(f"        this.z2_{i} = this.z1_{i};\n")
        stream.write(f"        this.z1_{i} = x{i};\n")

    stream.write("\n        return output;\n")
    stream.write("    }\n\n")

    stream.write("    public void reset() {\n")
    for i in range(sections):
        stream.write(f"        this.z1_{i} = 0.0;\n")
        stream.write(f"        this.z2_{i} = 0.0;\n")
    stream.write("    }\n")
    stream.write("}\n\n")

    # Add simple example usage
    stream.write("// Example usage:\n")
    stream.write("// Single channel:\n")
    stream.write(f"// {name} filter = new {name}();\n")
    stream.write("// filter.reset();\n")
    stream.write("// double filtered_output = filter.process(sample);\n")
    stream.write("// \n")
    stream.write("// Multi-channel (3 channels):\n")
    stream.write(f"// {name}[] filters = new {name}[3];  // One filter per channel\n")
    stream.write("// for(int i = 0; i < 3; i++) filters[i] = new {name}();\n")
    stream.write("// double filtered_1 = filters[0].process(raw1);\n")
    stream.write("// double filtered_2 = filters[1].process(raw2);\n")
    stream.write("// double filtered_3 = filters[2].process(raw3);\n")

################################################################
def emit_filter_code(stream, name, sos, filter_type, fs, freqs, order, language):
    """Emit filter code in the specified language"""
    language_map = {
        'python': emit_python_filter,
        'c++': emit_cpp_filter,
        'javascript': emit_javascript_filter,
        'typescript': emit_typescript_filter,
        'java': emit_java_filter
    }

    emit_func = language_map.get(language.lower(), emit_python_filter)
    emit_func(stream, name, sos, filter_type, fs, freqs, order)

################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Generate complete filter implementations for Butterworth IIR digital filters.""",
                                   formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog=usage_guide)

    parser.add_argument('--type', default='lowpass', type=str,
                       choices = ['lowpass', 'highpass', 'bandpass', 'bandstop'],
                       help = 'Filter type: lowpass, highpass, bandpass, bandstop (default lowpass).')

    parser.add_argument('--rate', default=10, type=float, help = 'Sampling frequency in Hz (default 10).')

    parser.add_argument('--order', default=4, type=int, help = 'Filter order (default 4).')

    parser.add_argument('--freq', default=1.0, type=float, help = 'Critical frequency (default 1.0 Hz).')

    parser.add_argument('--width', default=1.0, type=float, help = 'Bandwidth (for bandpass or bandstop) (default 1.0 Hz).')

    parser.add_argument('--name', type=str, help = 'Name of filter class/function.')

    parser.add_argument('--language', default='python', type=str,
                       choices=['python', 'c++', 'javascript', 'typescript', 'java'],
                       help='Output language (default python).')

    parser.add_argument('--out', type=str, help='Path of output file for filter code.')

    parser.add_argument('--plot', type=str, help='Path of optional plot output image file.')

    args = parser.parse_args()

    if args.type == 'lowpass':
        freqs = args.freq
        funcname = 'LowpassFilter' if args.name is None else args.name

    elif args.type == 'highpass':
        freqs = args.freq
        funcname = 'HighpassFilter' if args.name is None else args.name

    elif args.type == 'bandpass':
        freqs = [args.freq - 0.5*args.width, args.freq + 0.5*args.width]
        funcname = 'BandpassFilter' if args.name is None else args.name

    elif args.type == 'bandstop':
        freqs = [args.freq - 0.5*args.width, args.freq + 0.5*args.width]
        funcname = 'BandstopFilter' if args.name is None else args.name

    # Generate a Butterworth filter as a cascaded series of second-order digital
    # filters (second-order sections aka biquad).
    sos = scipy.signal.butter(N=args.order, Wn=freqs, btype=args.type, analog=False, output='sos', fs=args.rate)

    logging.debug("SOS filter: %s", sos)

    # Determine file extension based on language
    extensions = {
        'python': '.py',
        'c++': '.cpp',
        'javascript': '.js',
        'typescript': '.ts',
        'java': '.java'
    }

    if args.out is None:
        filename = args.type + extensions.get(args.language, '.py')
    else:
        filename = args.out

    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    with open(filename, "w") as stream:
        emit_filter_code(stream, funcname, sos, args.type, args.rate, freqs, args.order, args.language)

    print(f"Generated {args.language} filter: {filename}")

    if args.plot is not None:
        printable_type = type_print_form[args.type]
        make_plots(args.plot, sos, args.rate, args.order, freqs, printable_type)

################################################################
