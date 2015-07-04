#ifndef JSON_SERIALIZER_HPP
#define JSON_SERIALIZER_HPP

#include <vector>
#include "jansson.h"
#include "winjunk.hpp"

using namespace cv;
using namespace std;

namespace firesight {

typedef class JSONSerializer {
	private:
		int	flags;
		int precision;
		int indent;
	public:
		JSONSerializer(int flags = JSON_COMPACT|JSON_PRESERVE_ORDER)
			: flags(flags), precision(0), indent(0) {
			setIndent(2);
			setPrecision(6);
		}
	public:
		void setIndent(int value) {
			flags &= ~(JSON_INDENT(this->indent));
			this->indent = value;
			flags |= JSON_INDENT(value);
		}
		int	getIndent() {
			return indent;
		}
	public:
		void setPrecision(int value) {
			flags &= ~(JSON_REAL_PRECISION(this->precision));
			this->precision = value;
			flags |= JSON_REAL_PRECISION(value);
		}
		int getPrecision() {
			return precision;
		}
	public:
		string serialize(json_t *pnode) {
			string result;
			char * bytes = json_dumps(pnode, flags);
			if (bytes) {
				result = bytes;
				free(bytes);
			} else {
				result = "(NOMEM)";
			}
			return result;
		}
} JSONSerializer;

extern JSONSerializer defaultSerializer;

} // namespace firesight

#endif
