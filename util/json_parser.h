#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"
#include "rapidjson/filereadstream.h"


using namespace rapidjson;

namespace mlp {

	typedef const char* ConstCPtr;

	class JsonParser
	{
	public:
		explicit JsonParser(ConstCPtr netfile_path) {
			_parser(netfile_path);
		}

		inline int get_output_num(ConstCPtr type_str) const {
			assert(((type_str == "Data") || (type_str == "Loss")));
			assert(document.HasMember(type_str));
			assert(document[type_str].IsObject());
			assert(document[type_str].HasMember("output_num"));
			assert(document[type_str]["output_num"].IsInt());
			return document[type_str]["output_num"].GetInt();
		}

		inline ConstCPtr get_type(ConstCPtr type_str) const {
			assert(document.HasMember(type_str));
			assert(document[type_str].IsObject());
			assert(document[type_str].HasMember("type"));
			assert(document[type_str]["type"].IsString());
			return document[type_str]["type"].GetString();
		}

		inline ConstCPtr get_data_path() const {
			assert(document.HasMember("Data"));
			assert(document["Data"].IsObject());
			assert(document["Data"].HasMember("file_path"));
			assert(document["Data"]["file_path"].IsString());
			return document["Data"]["file_path"].GetString();
		}

		inline SizeType get_hidden_num() const {
			assert(document.HasMember("Inner"));
			assert(document["Inner"].IsObject());
			assert(document["Inner"].HasMember("hidden_num"));
			assert(document["Inner"]["hidden_num"].IsInt());
			return document["Inner"]["hidden_num"].GetInt();
		}

		inline void get_neuron_nums(std::vector<int>& neuron_nums) const {
			assert(document.HasMember("Inner"));
			assert(document["Inner"].IsObject());
			assert(document["Inner"].HasMember("neuron_num"));
			assert(document["Inner"]["neuron_num"].IsArray());
			SizeType arr_size = document["Inner"]["neuron_num"].Size();
			assert(get_hidden_num() == arr_size);
			for (Value::ConstValueIterator itr = document["Inner"]["neuron_num"].Begin();
				itr != document["Inner"]["neuron_num"].End(); ++itr) {
				neuron_nums.push_back(itr->GetInt());
			}
		}
		
		inline ConstCPtr get_net_name() const {
			assert(document.HasMember("Name"));
			assert(document["Name"].IsString());
			return document["Name"].GetString();
		}

	private:
		inline void _parser(ConstCPtr netfile_path) {
			std::ifstream ifs(netfile_path);
			IStreamWrapper isw(ifs);
			document.ParseStream(isw);
			assert(document.IsObject());
			/*
			FILE* fp = nullptr;
			fopen_s(&fp, netfile_path, "rb");
			char readBuffer[65536];
			FileReadStream is(fp, readBuffer, sizeof(readBuffer));
			document.ParseStream(is);
			fclose(fp);
			*/
		}

	private:
		rapidjson::Document document;
	};

}
