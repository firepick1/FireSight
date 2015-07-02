#ifndef FIRESIGHT_HPP
#define FIRESIGHT_HPP




#if __amd64__ || __x86_64__ || _WIN64 || _M_X64
#define FIRESIGHT_64_BIT
#define FIRESIGHT_PLATFORM_BITS 64
#else
#define FIRESIGHT_32_BIT
#define FIRESIGHT_PLATFORM_BITS 32
#endif

namespace firesight {



} // namespace firesight

#endif
