#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <map>
#include <iterator>
#include <filesystem>
#include <numeric>
#include <optional>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include "copyright.h"

#ifdef STORMM_USE_HPC
#  include "Accelerator/gpu_details.h"
#  include "Accelerator/hpc_config.h"
#  include "Accelerator/hybrid.h"
#endif

#include "Namelists/command_line_parser.h"
#include "Namelists/namelist_emulator.h"

#include "Potential/static_exclusionmask.h"
#include "Reporting/error_format.h"
#include "Sampling/gcmc_sampler.h"
#include "Structure/structure_enumerators.h"
#include "Topology/atomgraph.h"
#include "Topology/atomgraph_enumerators.h"
#include "Trajectory/phasespace.h"
#include "Trajectory/thermostat.h"
#include "Trajectory/trajectory_enumerators.h"
#include "Chemistry/znumber.h"

using namespace stormm;
using namespace stormm::card;
using namespace stormm::energy;
using namespace stormm::errors;
using namespace stormm::namelist;
using namespace stormm::sampling;
using namespace stormm::structure;
using namespace stormm::topology;
using namespace stormm::trajectory;
using namespace stormm::chemistry;

namespace {

using clock = std::chrono::high_resolution_clock;

GhostMoleculeMetadata buildSystemWithGhosts(const std::string &protein_prmtop,
                                            const std::string &protein_inpcrd,
                                            const std::string &fragment_prmtop,
                                            const std::string &fragment_inpcrd,
                                            const int n_ghosts,
                                            AtomGraph *combined_topology,
                                            PhaseSpace *phase_space,
                                            std::vector<double> *box_dims)
{
  AtomGraph fragment(fragment_prmtop, ExceptionResponse::WARN);

  AtomGraph base_topology;
  int base_molecule_count = 0;
  int base_atom_count = 0;

  if (!protein_prmtop.empty()) {
    base_topology = AtomGraph(protein_prmtop, ExceptionResponse::WARN);
    base_molecule_count = base_topology.getMoleculeCount();
    base_atom_count = base_topology.getAtomCount();
  }

  *combined_topology = buildTopologyWithGhosts(base_topology, fragment, n_ghosts);

  GhostMoleculeMetadata ghost_metadata =
      identifyGhostMolecules(*combined_topology, base_molecule_count, n_ghosts);

  if (ghost_metadata.n_ghost_molecules != n_ghosts) {
    rtErr("Combined topology reports " + std::to_string(ghost_metadata.n_ghost_molecules) +
          " ghost molecules but " + std::to_string(n_ghosts) +
          " were requested. Check fragment topology consistency.",
          "gcmc_hybrid_runner");
  }
  if (ghost_metadata.base_atom_count != base_atom_count) {
    rtErr("Combined topology expected " + std::to_string(base_atom_count) +
          " protein atoms but found " + std::to_string(ghost_metadata.base_atom_count) +
          ". Verify that the protein prmtop / inpcrd pair aligns.",
          "gcmc_hybrid_runner");
  }

  const int total_atoms = combined_topology->getAtomCount();
  std::vector<double> xcrd(total_atoms, 0.0);
  std::vector<double> ycrd(total_atoms, 0.0);
  std::vector<double> zcrd(total_atoms, 0.0);

  PhaseSpace fragment_ps(fragment_inpcrd, CoordinateFileKind::AMBER_INPCRD);
  const PhaseSpaceReader frag_psr = fragment_ps.data();
  const int fragment_natoms = fragment.getAtomCount();

  if (fragment_ps.getAtomCount() != fragment_natoms) {
    rtErr("Fragment coordinate count mismatch", "gcmc_hybrid_runner");
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  if (!protein_inpcrd.empty()) {
    PhaseSpace protein_ps(protein_inpcrd, CoordinateFileKind::AMBER_INPCRD);
    const PhaseSpaceReader protein_psr = protein_ps.data();

    if (protein_ps.getAtomCount() != ghost_metadata.base_atom_count) {
      rtErr("Protein coordinate count mismatch", "gcmc_hybrid_runner");
    }

    for (int i = 0; i < ghost_metadata.base_atom_count; i++) {
      xcrd[i] = protein_psr.xcrd[i];
      ycrd[i] = protein_psr.ycrd[i];
      zcrd[i] = protein_psr.zcrd[i];
    }

    const double box_threshold = 1.0;
    if (protein_psr.boxdim[0] > box_threshold &&
        protein_psr.boxdim[1] > box_threshold &&
        protein_psr.boxdim[2] > box_threshold)
    {
      for (int i = 0; i < 6; i++) {
        (*box_dims)[i] = protein_psr.boxdim[i];
      }
    }
    else {
      const double default_box_size = 25.0;
      (*box_dims)[0] = default_box_size;
      (*box_dims)[1] = default_box_size;
      (*box_dims)[2] = default_box_size;
      (*box_dims)[3] = 90.0;
      (*box_dims)[4] = 90.0;
      (*box_dims)[5] = 90.0;
      std::cout << "  Note: protein input lacks box dimensions, using "
                << default_box_size << " Å cubic box\n";
    }
  }
  else {
    const double default_box_size = 25.0;
    (*box_dims)[0] = default_box_size;
    (*box_dims)[1] = default_box_size;
    (*box_dims)[2] = default_box_size;
    (*box_dims)[3] = 90.0;
    (*box_dims)[4] = 90.0;
    (*box_dims)[5] = 90.0;
    std::cout << "  No protein coordinates supplied; using "
              << default_box_size << " Å cubic box\n";
  }

  std::uniform_real_distribution<double> dis_x(0.0, (*box_dims)[0]);
  std::uniform_real_distribution<double> dis_y(0.0, (*box_dims)[1]);
  std::uniform_real_distribution<double> dis_z(0.0, (*box_dims)[2]);

  double frag_com_x = 0.0, frag_com_y = 0.0, frag_com_z = 0.0;
  for (int i = 0; i < fragment_natoms; i++) {
    frag_com_x += frag_psr.xcrd[i];
    frag_com_y += frag_psr.ycrd[i];
    frag_com_z += frag_psr.zcrd[i];
  }
  frag_com_x /= fragment_natoms;
  frag_com_y /= fragment_natoms;
  frag_com_z /= fragment_natoms;

  for (int ghost_idx = 0; ghost_idx < ghost_metadata.n_ghost_molecules; ghost_idx++) {
    const double rand_x = dis_x(gen);
    const double rand_y = dis_y(gen);
    const double rand_z = dis_z(gen);
    const int atom_offset = ghost_metadata.base_atom_count + (ghost_idx * fragment_natoms);
    for (int i = 0; i < fragment_natoms; i++) {
      xcrd[atom_offset + i] = frag_psr.xcrd[i] - frag_com_x + rand_x;
      ycrd[atom_offset + i] = frag_psr.ycrd[i] - frag_com_y + rand_y;
      zcrd[atom_offset + i] = frag_psr.zcrd[i] - frag_com_z + rand_z;
    }
  }

  *phase_space = PhaseSpace(total_atoms);
  phase_space->fill(xcrd, ycrd, zcrd,
                    TrajectoryKind::POSITIONS, CoordinateCycle::WHITE, 0, *box_dims);

  return ghost_metadata;
}

void zeroProteinMasses(AtomGraph *combined_topology, int protein_atom_count) {
  if (protein_atom_count <= 0) {
    return;
  }
  ChemicalDetailsKit cdk = combined_topology->getChemicalDetailsKit();
  double* masses = const_cast<double*>(cdk.masses);
  float* sp_masses = const_cast<float*>(cdk.sp_masses);
  double* inv_masses = const_cast<double*>(cdk.inv_masses);
  float* sp_inv_masses = const_cast<float*>(cdk.sp_inv_masses);

  for (int i = 0; i < protein_atom_count; i++) {
    masses[i] = 0.0;
    sp_masses[i] = 0.0f;
    inv_masses[i] = 0.0;
    sp_inv_masses[i] = 0.0f;
  }

  combined_topology->upload();
}

void zeroProteinVelocities(PhaseSpace *phase_space, int protein_atom_count) {
  if (protein_atom_count <= 0) {
    return;
  }
  PhaseSpaceWriter psw = phase_space->data();
  for (int i = 0; i < protein_atom_count; i++) {
    psw.xvel[i] = 0.0;
    psw.yvel[i] = 0.0;
    psw.zvel[i] = 0.0;
  }
}

std::string trimCopy(const std::string& input) {
  const size_t first = input.find_first_not_of(" \t\r\n");
  const size_t last  = input.find_last_not_of(" \t\r\n");
  if (first == std::string::npos || last == std::string::npos) {
    return std::string();
  }
  return input.substr(first, last - first + 1);
}

std::vector<std::string> tokenizeConfigFile(const std::filesystem::path& filename) {
  std::ifstream config(filename);
  if (!config.is_open()) {
    rtErr("Could not open configuration file: " + filename.string(), "gcmc_hybrid_runner");
  }

  std::vector<std::string> tokens;
  std::string line;
  while (std::getline(config, line)) {
    const size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = line.substr(0, comment_pos);
    }
    const std::string trimmed = trimCopy(line);
    if (trimmed.empty()) {
      continue;
    }

    std::istringstream iss(trimmed);
    std::string token;
    while (iss >> token) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

enum class JsonType {
  Null,
  Bool,
  Number,
  String,
  Array,
  Object
};

struct JsonValue {
  JsonType type;
  bool bool_value;
  double number_value;
  std::string string_value;
  std::vector<JsonValue> array_value;
  std::vector<std::pair<std::string, JsonValue>> object_value;

  JsonValue() : type(JsonType::Null), bool_value(false), number_value(0.0) {}
};

void skipJsonWhitespace(const std::string& text, size_t* pos) {
  while (*pos < text.size()) {
    const unsigned char c = static_cast<unsigned char>(text[*pos]);
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      (*pos)++;
    }
    else {
      break;
    }
  }
}

int parseHexDigit(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return 10 + (c - 'a');
  }
  if (c >= 'A' && c <= 'F') {
    return 10 + (c - 'A');
  }
  rtErr("Invalid hexadecimal digit '" + std::string(1, c) + "' in \\u escape sequence.",
        "gcmc_hybrid_runner");
  return 0;
}

std::string parseJsonStringLiteral(const std::string& text, size_t* pos) {
  if (*pos >= text.size() || text[*pos] != '"') {
    rtErr("Expected '\"' to begin JSON string.", "gcmc_hybrid_runner");
  }
  (*pos)++;
  std::string result;
  while (*pos < text.size()) {
    const char c = text[*pos];
    if (c == '"') {
      (*pos)++;
      return result;
    }
    if (c == '\\') {
      (*pos)++;
      if (*pos >= text.size()) {
        rtErr("Unterminated escape sequence in JSON string.", "gcmc_hybrid_runner");
      }
      const char esc = text[*pos];
      switch (esc) {
      case '"':
      case '\\':
      case '/':
        result.push_back(esc);
        break;
      case 'b':
        result.push_back('\b');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      case 'u': {
        if (*pos + 4 >= text.size()) {
          rtErr("Incomplete \\u escape in JSON string.", "gcmc_hybrid_runner");
        }
        int code_point = 0;
        for (int i = 1; i <= 4; i++) {
          code_point = (code_point << 4) + parseHexDigit(text[*pos + i]);
        }
        (*pos) += 4;
        if (code_point <= 0x7F) {
          result.push_back(static_cast<char>(code_point));
        }
        else if (code_point <= 0x7FF) {
          result.push_back(static_cast<char>(0xC0 | ((code_point >> 6) & 0x1F)));
          result.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        }
        else {
          result.push_back(static_cast<char>(0xE0 | ((code_point >> 12) & 0x0F)));
          result.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
          result.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
        }
        break;
      }
      default:
        rtErr("Unsupported escape character '\\" + std::string(1, esc) + "' in JSON string.",
              "gcmc_hybrid_runner");
      }
    }
    else {
      result.push_back(c);
    }
    (*pos)++;
  }
  rtErr("Unterminated JSON string literal.", "gcmc_hybrid_runner");
  return std::string();
}

JsonValue parseJsonValue(const std::string& text, size_t* pos);

JsonValue parseJsonArray(const std::string& text, size_t* pos) {
  if (*pos >= text.size() || text[*pos] != '[') {
    rtErr("Expected '[' to begin JSON array.", "gcmc_hybrid_runner");
  }
  (*pos)++;
  JsonValue result;
  result.type = JsonType::Array;
  skipJsonWhitespace(text, pos);
  if (*pos < text.size() && text[*pos] == ']') {
    (*pos)++;
    return result;
  }
  while (*pos < text.size()) {
    JsonValue element = parseJsonValue(text, pos);
    result.array_value.push_back(std::move(element));
    skipJsonWhitespace(text, pos);
    if (*pos >= text.size()) {
      break;
    }
    if (text[*pos] == ',') {
      (*pos)++;
      skipJsonWhitespace(text, pos);
      continue;
    }
    if (text[*pos] == ']') {
      (*pos)++;
      return result;
    }
    rtErr("Expected ',' or ']' inside JSON array.", "gcmc_hybrid_runner");
  }
  rtErr("Unterminated JSON array.", "gcmc_hybrid_runner");
  return result;
}

JsonValue parseJsonObject(const std::string& text, size_t* pos) {
  if (*pos >= text.size() || text[*pos] != '{') {
    rtErr("Expected '{' to begin JSON object.", "gcmc_hybrid_runner");
  }
  (*pos)++;
  JsonValue result;
  result.type = JsonType::Object;
  skipJsonWhitespace(text, pos);
  if (*pos < text.size() && text[*pos] == '}') {
    (*pos)++;
    return result;
  }
  while (*pos < text.size()) {
    skipJsonWhitespace(text, pos);
    const std::string key = parseJsonStringLiteral(text, pos);
    skipJsonWhitespace(text, pos);
    if (*pos >= text.size() || text[*pos] != ':') {
      rtErr("Expected ':' after key \"" + key + "\" in JSON object.", "gcmc_hybrid_runner");
    }
    (*pos)++;
    skipJsonWhitespace(text, pos);
    JsonValue value = parseJsonValue(text, pos);
    result.object_value.emplace_back(key, std::move(value));
    skipJsonWhitespace(text, pos);
    if (*pos >= text.size()) {
      break;
    }
    if (text[*pos] == ',') {
      (*pos)++;
      skipJsonWhitespace(text, pos);
      continue;
    }
    if (text[*pos] == '}') {
      (*pos)++;
      return result;
    }
    rtErr("Expected ',' or '}' inside JSON object.", "gcmc_hybrid_runner");
  }
  rtErr("Unterminated JSON object.", "gcmc_hybrid_runner");
  return result;
}

JsonValue parseJsonNumber(const std::string& text, size_t* pos) {
  size_t start = *pos;
  if (text[*pos] == '-') {
    (*pos)++;
  }
  if (*pos >= text.size() || (text[*pos] < '0' || text[*pos] > '9')) {
    rtErr("Invalid JSON number format.", "gcmc_hybrid_runner");
  }
  if (text[*pos] == '0') {
    (*pos)++;
  }
  else {
    while (*pos < text.size() && std::isdigit(static_cast<unsigned char>(text[*pos]))) {
      (*pos)++;
    }
  }
  if (*pos < text.size() && text[*pos] == '.') {
    (*pos)++;
    if (*pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[*pos]))) {
      rtErr("Invalid JSON number fractional part.", "gcmc_hybrid_runner");
    }
    while (*pos < text.size() && std::isdigit(static_cast<unsigned char>(text[*pos]))) {
      (*pos)++;
    }
  }
  if (*pos < text.size() && (text[*pos] == 'e' || text[*pos] == 'E')) {
    (*pos)++;
    if (*pos < text.size() && (text[*pos] == '+' || text[*pos] == '-')) {
      (*pos)++;
    }
    if (*pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[*pos]))) {
      rtErr("Invalid JSON number exponent.", "gcmc_hybrid_runner");
    }
    while (*pos < text.size() && std::isdigit(static_cast<unsigned char>(text[*pos]))) {
      (*pos)++;
    }
  }
  const std::string number_text = text.substr(start, *pos - start);
  JsonValue result;
  result.type = JsonType::Number;
  try {
    result.number_value = std::stod(number_text);
  }
  catch (const std::exception&) {
    rtErr("Unable to convert \"" + number_text + "\" to a floating-point value.",
          "gcmc_hybrid_runner");
  }
  return result;
}

JsonValue parseJsonValue(const std::string& text, size_t* pos) {
  skipJsonWhitespace(text, pos);
  if (*pos >= text.size()) {
    rtErr("Unexpected end of JSON content.", "gcmc_hybrid_runner");
  }
  const char c = text[*pos];
  if (c == '"') {
    JsonValue result;
    result.type = JsonType::String;
    result.string_value = parseJsonStringLiteral(text, pos);
    return result;
  }
  if (c == '{') {
    return parseJsonObject(text, pos);
  }
  if (c == '[') {
    return parseJsonArray(text, pos);
  }
  if (c == 't') {
    if (text.compare(*pos, 4, "true") != 0) {
      rtErr("Invalid literal in JSON (expected \"true\").", "gcmc_hybrid_runner");
    }
    (*pos) += 4;
    JsonValue result;
    result.type = JsonType::Bool;
    result.bool_value = true;
    return result;
  }
  if (c == 'f') {
    if (text.compare(*pos, 5, "false") != 0) {
      rtErr("Invalid literal in JSON (expected \"false\").", "gcmc_hybrid_runner");
    }
    (*pos) += 5;
    JsonValue result;
    result.type = JsonType::Bool;
    result.bool_value = false;
    return result;
  }
  if (c == 'n') {
    if (text.compare(*pos, 4, "null") != 0) {
      rtErr("Invalid literal in JSON (expected \"null\").", "gcmc_hybrid_runner");
    }
    (*pos) += 4;
    return JsonValue();
  }
  if (c == '-' || (c >= '0' && c <= '9')) {
    return parseJsonNumber(text, pos);
  }
  rtErr(std::string("Unexpected character '") + c + "' while parsing JSON value.",
        "gcmc_hybrid_runner");
  return JsonValue();
}

JsonValue parseJsonDocument(const std::string& text) {
  size_t pos = 0;
  JsonValue result = parseJsonValue(text, &pos);
  skipJsonWhitespace(text, &pos);
  if (pos != text.size()) {
    rtErr("Extra data found after parsing JSON document.", "gcmc_hybrid_runner");
  }
  return result;
}

std::string toCliFlag(const std::string& key) {
  if (key.empty()) {
    rtErr("An empty string is not a valid CLI flag name in JSON configuration.",
          "gcmc_hybrid_runner");
  }
  if (key[0] == '-') {
    return key;
  }
  if (key.size() == 1) {
    return "-" + key;
  }
  return "--" + key;
}

std::string numberToString(double value) {
  if (std::isfinite(value)) {
    const double rounded = std::round(value);
    if (std::fabs(value - rounded) < 1.0e-12) {
      return std::to_string(static_cast<long long>(rounded));
    }
  }
  std::ostringstream oss;
  oss << std::setprecision(15) << value;
  return oss.str();
}

void appendJsonPrimitiveToken(const std::string& flag, const JsonValue& value,
                              std::vector<std::string>* tokens) {
  switch (value.type) {
  case JsonType::Bool:
    tokens->push_back(flag);
    tokens->push_back(value.bool_value ? "true" : "false");
    break;
  case JsonType::Number:
    tokens->push_back(flag);
    tokens->push_back(numberToString(value.number_value));
    break;
  case JsonType::String:
    tokens->push_back(flag);
    tokens->push_back(value.string_value);
    break;
  default:
    rtErr("JSON value assigned to " + flag +
          " must be a string, number, or boolean (or an array of those).",
          "gcmc_hybrid_runner");
  }
}

void appendJsonConfigTokens(const JsonValue& json,
                            std::vector<std::string>* tokens) {
  if (json.type == JsonType::Array) {
    for (const JsonValue& element : json.array_value) {
      if (element.type != JsonType::String) {
        rtErr("The root JSON array must contain only strings representing CLI arguments.",
              "gcmc_hybrid_runner");
      }
      tokens->push_back(element.string_value);
    }
    return;
  }
  if (json.type != JsonType::Object) {
    rtErr("Top-level JSON configuration must be an object or array.", "gcmc_hybrid_runner");
  }
  for (const auto& entry : json.object_value) {
    const std::string& key = entry.first;
    const JsonValue& value = entry.second;
    if (key == "args" || key == "--args") {
      if (value.type != JsonType::Array) {
        rtErr("\"args\" in JSON configuration must be an array of strings.",
              "gcmc_hybrid_runner");
      }
      for (const JsonValue& element : value.array_value) {
        if (element.type != JsonType::String) {
          rtErr("\"args\" array in JSON configuration must contain only strings.",
                "gcmc_hybrid_runner");
        }
        tokens->push_back(element.string_value);
      }
      continue;
    }
    const std::string flag = toCliFlag(key);
    if (value.type == JsonType::Array) {
      for (const JsonValue& item : value.array_value) {
        if (item.type == JsonType::Array || item.type == JsonType::Object) {
          rtErr("Nested arrays or objects are not supported for \"" + flag + "\".",
                "gcmc_hybrid_runner");
        }
        appendJsonPrimitiveToken(flag, item, tokens);
      }
      continue;
    }
    if (value.type == JsonType::Object) {
      rtErr("Nested JSON objects are not supported for key \"" + key +
            "\". Use primitive values, arrays, or the \"args\" field.",
            "gcmc_hybrid_runner");
    }
    if (value.type == JsonType::Null) {
      rtErr("Null is not a supported value type for \"" + flag + "\".",
            "gcmc_hybrid_runner");
    }
    appendJsonPrimitiveToken(flag, value, tokens);
  }
}

std::vector<std::string> tokenizeJsonConfig(const std::filesystem::path& filename) {
  std::ifstream config(filename);
  if (!config.is_open()) {
    rtErr("Could not open JSON configuration file: " + filename.string(),
          "gcmc_hybrid_runner");
  }
  const std::string content((std::istreambuf_iterator<char>(config)),
                            std::istreambuf_iterator<char>());
  const JsonValue root = parseJsonDocument(content);
  std::vector<std::string> tokens;
  tokens.reserve(32);
  appendJsonConfigTokens(root, &tokens);
  return tokens;
}

void expandArgumentsRecursive(const std::vector<std::string>& input,
                              std::vector<std::string>* output,
                              const std::filesystem::path& base_dir,
                              std::unordered_set<std::string>* visited_configs) {
  for (size_t idx = 0; idx < input.size(); idx++) {
    const std::string& token = input[idx];
    if (token == "--config" || token == "--config-json") {
      if (idx + 1 >= input.size()) {
        rtErr("Expected a configuration file path after " + token + ".", "gcmc_hybrid_runner");
      }
      const std::filesystem::path config_path_in = input[idx + 1];
      std::filesystem::path resolved_path = config_path_in;
      if (!resolved_path.is_absolute()) {
        resolved_path = std::filesystem::absolute(base_dir / config_path_in);
      }
      if (!std::filesystem::exists(resolved_path)) {
        rtErr("Configuration file " + resolved_path.string() + " does not exist.",
              "gcmc_hybrid_runner");
      }
      resolved_path = std::filesystem::weakly_canonical(resolved_path);
      const std::string normalized = resolved_path.lexically_normal().string();
      if (!visited_configs->insert(normalized).second) {
        rtErr("Configuration file recursion detected at " + normalized,
              "gcmc_hybrid_runner");
      }
      std::string extension = resolved_path.extension().string();
      std::string extension_lower = extension;
      for (char& c : extension_lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      }
      const bool force_json = (token == "--config-json");
      const bool is_json = force_json || (extension_lower == ".json");
      const std::vector<std::string> cfg_tokens =
          (is_json) ? tokenizeJsonConfig(resolved_path)
                    : tokenizeConfigFile(resolved_path);
      expandArgumentsRecursive(cfg_tokens, output, resolved_path.parent_path(),
                               visited_configs);
      visited_configs->erase(normalized);
      idx++;
    }
    else {
      output->push_back(token);
    }
  }
}

std::vector<std::string> expandArguments(const std::vector<std::string>& raw_args) {
  std::vector<std::string> expanded;
  expanded.reserve(raw_args.size() + 16);
  std::unordered_set<std::string> visited_configs;
  expandArgumentsRecursive(raw_args, &expanded, std::filesystem::current_path(),
                           &visited_configs);
  return expanded;
}

bool parseBoolString(const std::string& input, bool* value_out) {
  std::string lowered;
  lowered.reserve(input.size());
  for (char c : input) {
    lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
    *value_out = true;
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
    *value_out = false;
    return true;
  }
  return false;
}

bool readEnvBoolOverride(const char* env_value, bool* value_out) {
  if (env_value == nullptr) {
    return false;
  }
  const std::string raw(env_value);
  bool parsed = false;
  if (!parseBoolString(raw, &parsed)) {
    rtWarn("Environment variable STORMM_LOG_MEMORY has unrecognized value '" + raw +
           "'. Expected one of {0,1,true,false,on,off,yes,no}. Using default.",
           "gcmc_hybrid_runner");
    return false;
  }
  *value_out = parsed;
  return true;
}

std::filesystem::path ensureOutputDirectory(const std::string& output_dir) {
  if (output_dir.empty()) {
    return std::filesystem::current_path();
  }
  const std::filesystem::path dir_path(output_dir);
  std::error_code create_err;
  std::filesystem::create_directories(dir_path, create_err);
  if (create_err) {
    rtErr("Failed to create output directory '" + dir_path.string() +
          "': " + create_err.message(), "gcmc_hybrid_runner");
  }
  return dir_path;
}

std::filesystem::path combinePrefix(const std::filesystem::path& directory,
                                    const std::string& prefix) {
  if (prefix.empty()) {
    return directory / "gcmc_hybrid_output";
  }
  return directory / prefix;
}

void validateMCSettings(double translation, double rotation, int frequency) {
  if (translation <= 0.0 && rotation <= 0.0) {
    std::cout << "  Warning: MC movers disabled (zero translation and rotation amplitudes).\n";
  }
  else {
    if (translation <= 0.0) {
      std::cout << "  Warning: MC translation amplitude <= 0.0; translation moves disabled.\n";
    }
    if (rotation <= 0.0) {
      std::cout << "  Warning: MC rotation amplitude <= 0.0; rotation moves disabled.\n";
    }
  }
  if (frequency <= 0) {
    std::cout << "  Warning: MC frequency <= 0; MC movers will not be attempted.\n";
  }
}

void validateSimulationSettings(int n_moves, double temperature, double timestep,
                                int md_steps, int n_ghosts) {
  if (n_moves <= 0) {
    rtErr("The number of GCMC cycles must be positive (--moves / -n).",
          "gcmc_hybrid_runner");
  }
  if (temperature <= 0.0) {
    rtErr("Simulation temperature (--temp) must be greater than 0 K.", "gcmc_hybrid_runner");
  }
  if (timestep <= 0.0) {
    rtErr("MD timestep (--timestep) must be greater than 0 fs.", "gcmc_hybrid_runner");
  }
  if (md_steps < 0) {
    rtErr("MD step count (--md-steps) must be non-negative.", "gcmc_hybrid_runner");
  }
  if (n_ghosts <= 0) {
    rtErr("Number of ghost templates (--nghost) must be positive.", "gcmc_hybrid_runner");
  }
}

void verifyInputFile(const std::string& path, const std::string& label) {
  if (path.empty()) {
    rtErr("A required " + label + " path was not provided.", "gcmc_hybrid_runner");
  }
  if (!std::filesystem::exists(path)) {
    rtErr(label + " file '" + path + "' was not found.", "gcmc_hybrid_runner");
  }
}

std::string describeAnnealingStage(AnnealingStage stage) {
  switch (stage) {
  case AnnealingStage::DISCOVERY:
    return "Discovery";
  case AnnealingStage::COARSE:
    return "Coarse Equilibration";
  case AnnealingStage::FINE:
    return "Fine Annealing";
  case AnnealingStage::PRODUCTION:
    return "Production";
  }
  return "Unknown";
}

void writePdbSnapshot(const std::filesystem::path& file_path,
                      const AtomGraph& topology,
                      const PhaseSpace& phase_space,
                      const std::vector<int>& atom_indices,
                      const std::string& empty_message) {
  std::filesystem::path parent_dir = file_path.parent_path();
  if (!parent_dir.empty()) {
    std::error_code dir_err;
    std::filesystem::create_directories(parent_dir, dir_err);
    if (dir_err) {
      rtErr("Unable to create directory '" + parent_dir.string() + "' for PDB output: " +
            dir_err.message(), "writePdbSnapshot");
    }
  }

  std::ofstream pdb_file(file_path, std::ios::trunc);
  if (!pdb_file.is_open()) {
    rtErr("Unable to open PDB file '" + file_path.string() + "' for writing.",
          "writePdbSnapshot");
  }

  const PhaseSpaceReader psr = phase_space.data();
  pdb_file << std::fixed << std::setprecision(3);
  pdb_file << "CRYST1"
           << std::setw(9) << psr.boxdim[0]
           << std::setw(9) << psr.boxdim[1]
           << std::setw(9) << psr.boxdim[2]
           << std::setw(7) << psr.boxdim[3]
           << std::setw(7) << psr.boxdim[4]
           << std::setw(7) << psr.boxdim[5]
           << " P 1           1\n";

  if (atom_indices.empty()) {
    pdb_file << "REMARK   " << empty_message << "\n";
    pdb_file << "END\n";
    return;
  }

  const ChemicalDetailsKit cdk = topology.getChemicalDetailsKit();
  int serial = 1;
  for (const int atom_idx : atom_indices) {
    const double x = psr.xcrd[atom_idx];
    const double y = psr.ycrd[atom_idx];
    const double z = psr.zcrd[atom_idx];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    const int res_idx = topology.getResidueIndex(atom_idx);
    const char* atom_name_ptr = reinterpret_cast<const char*>(cdk.atom_names + atom_idx);
    const char* res_name_ptr  = reinterpret_cast<const char*>(cdk.res_names + res_idx);
    const char2 element_pair  = zNumberToSymbol(cdk.z_numbers[atom_idx]);
    char element_buf[3] = {element_pair.x, element_pair.y, '\0'};
    std::string element = element_buf;
    if (!element.empty()) {
      const size_t last_non_space = element.find_last_not_of(' ');
      if (last_non_space != std::string::npos) {
        element.erase(last_non_space + 1);
      }
      else {
        element.clear();
      }
    }

    pdb_file << "ATOM  "
             << std::setw(5) << serial
             << " "
             << std::left << std::setw(4) << std::string(atom_name_ptr, 4) << std::right
             << " "
             << std::left << std::setw(3) << std::string(res_name_ptr, 3) << std::right
             << " A"
             << std::setw(4) << (res_idx + 1)
             << "    "
             << std::setw(8) << x
             << std::setw(8) << y
             << std::setw(8) << z
             << "  1.00  0.00          "
             << std::right << std::setw(2) << element
             << " \n";
    serial++;
  }
  if (serial == 1) {
    pdb_file << "REMARK   " << empty_message << "\n";
  }
  pdb_file << "END\n";
}

} // namespace

//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {
#ifdef STORMM_USE_HPC
  const HpcConfig gpu_config(ExceptionResponse::WARN);
  const std::vector<int> my_gpus = gpu_config.getGpuDevice(1);
  const GpuDetails gpu = (my_gpus.empty()) ? null_gpu : gpu_config.getGpuInfo(my_gpus[0]);
  if (!my_gpus.empty()) {
    const Hybrid<int> array_to_trigger_gpu_mapping(1);
  }
#else
  const GpuDetails gpu = null_gpu;
#endif

  std::vector<std::string> raw_args;
  raw_args.reserve(static_cast<size_t>(std::max(0, argc - 1)));
  for (int i = 1; i < argc; i++) {
    raw_args.emplace_back(argv[i]);
  }
  const std::vector<std::string> expanded_args = expandArguments(raw_args);

  std::vector<const char*> argv_expanded;
  argv_expanded.reserve(expanded_args.size() + 1);
  argv_expanded.push_back(argv[0]);
  for (const std::string& token : expanded_args) {
    argv_expanded.push_back(token.c_str());
  }
  const int argc_expanded = static_cast<int>(argv_expanded.size());

  CommandLineParser clip("gcmc_hybrid.stormm",
                        "Hybrid GPU GCMC + MC sampler for protein / fragment systems");
  clip.addStandardApplicationInputs({ "-p", "-c", "-o" });

  NamelistEmulator* t_nml = clip.getNamelistPointer();

  t_nml->addKeyword("--fragment-prmtop", NamelistType::STRING, "");
  t_nml->addKeyword("--fragment-inpcrd", NamelistType::STRING, "");
  t_nml->addHelp("--fragment-prmtop", "Fragment topology file (required)");
  t_nml->addHelp("--fragment-inpcrd", "Fragment coordinate file (required)");

  t_nml->addKeyword("--nghost", NamelistType::INTEGER, "1000");
  t_nml->addHelp("--nghost", "Number of ghost templates for the fragment (default: 1000)");

  t_nml->addKeyword("--moves", NamelistType::INTEGER, "1000");
  t_nml->addKeyword("-n", NamelistType::INTEGER, "1000");
  t_nml->addHelp("--moves", "Number of GCMC cycles");
  t_nml->addHelp("-n", "Alias for --moves");

  t_nml->addKeyword("--temp", NamelistType::REAL, "300.0");
  t_nml->addHelp("--temp", "Simulation temperature in Kelvin");

  t_nml->addKeyword("--bvalue", NamelistType::REAL, "5.0");
  t_nml->addKeyword("-b", NamelistType::REAL, "5.0");
  t_nml->addHelp("--bvalue", "Adams B parameter");
  t_nml->addHelp("-b", "Alias for --bvalue");

  t_nml->addKeyword("--mu-ex", NamelistType::REAL, "0.0");
  t_nml->addKeyword("--standard-volume", NamelistType::REAL, "30.0");

  t_nml->addKeyword("--adaptive-b", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--adaptive-b", "Enable three-stage adaptive Adams-B control");
  t_nml->addKeyword("--stage1-moves", NamelistType::INTEGER, "300");
  t_nml->addKeyword("--stage2-moves", NamelistType::INTEGER, "300");
  t_nml->addKeyword("--stage3-moves", NamelistType::INTEGER, "300");
  t_nml->addKeyword("--b-discovery", NamelistType::REAL, "8.0");
  t_nml->addKeyword("--target-occupancy", NamelistType::REAL, "0.5");
  t_nml->addKeyword("--coarse-rate", NamelistType::REAL, "0.5");
  t_nml->addKeyword("--fine-rate", NamelistType::REAL, "0.25");
  t_nml->addKeyword("--b-min", NamelistType::REAL, "-5.0");
  t_nml->addKeyword("--b-max", NamelistType::REAL, "10.0");

  t_nml->addKeyword("--timestep", NamelistType::REAL, "2.0");
  t_nml->addKeyword("--md-steps", NamelistType::INTEGER, "50");

  t_nml->addKeyword("--mc-translation", NamelistType::REAL, "1.0");
  t_nml->addKeyword("--mc-rotation", NamelistType::REAL, "30.0");
  t_nml->addKeyword("--mc-frequency", NamelistType::INTEGER, "5");
  t_nml->addHelp("--mc-translation", "Max translation displacement in Å for MC moves");
  t_nml->addHelp("--mc-rotation", "Max rotation angle in degrees for MC moves");
  t_nml->addHelp("--mc-frequency", "Number of MC attempts per GCMC cycle");
  t_nml->addKeyword("--output-dir", NamelistType::STRING, "");
  t_nml->addHelp("--output-dir", "Directory for output logs (created if missing)");
  t_nml->addKeyword("--log-memory", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--log-memory",
                 "Record GPU / RSS memory telemetry each cycle (overridable with "
                 "STORMM_LOG_MEMORY env var)");
  t_nml->activateBool("--log-memory");
  t_nml->addKeyword("--final-pdb", NamelistType::STRING, "none");
  t_nml->addHelp("--final-pdb",
                 "Write a final PDB snapshot (use 'auto' for <output>-final.pdb)");
  t_nml->addKeyword("--final-pdb-active-only", NamelistType::BOOLEAN, "");
  t_nml->addHelp("--final-pdb-active-only",
                 "When writing the final PDB, include only active fragments");
  t_nml->addKeyword("--constant-b-pdb-prefix", NamelistType::STRING, "");
  t_nml->addHelp("--constant-b-pdb-prefix",
                 "When adaptive B is disabled, dump the first 100 cycles as PDBs using this prefix");

  clip.parseUserInput(argc_expanded, argv_expanded.data());

  const std::string protein_prmtop = t_nml->getStringValue("-p");
  const std::string protein_inpcrd = t_nml->getStringValue("-c");
  std::string output_prefix = t_nml->getStringValue("-o");
  const std::string fragment_prmtop = t_nml->getStringValue("--fragment-prmtop");
  const std::string fragment_inpcrd = t_nml->getStringValue("--fragment-inpcrd");
  const std::string output_dir_cli = t_nml->getStringValue("--output-dir");
  const std::string final_pdb_option = t_nml->getStringValue("--final-pdb");
  const bool final_pdb_active_only = t_nml->getBoolValue("--final-pdb-active-only");

  std::string constant_b_pdb_prefix_cli;
  const InputStatus constant_b_status =
      t_nml->getKeywordStatus("--constant-b-pdb-prefix");
  if (constant_b_status != InputStatus::MISSING) {
    constant_b_pdb_prefix_cli = t_nml->getStringValue("--constant-b-pdb-prefix");
  }

  verifyInputFile(protein_prmtop, "Protein topology");
  verifyInputFile(protein_inpcrd, "Protein coordinate");
  verifyInputFile(fragment_prmtop, "Fragment topology");
  verifyInputFile(fragment_inpcrd, "Fragment coordinate");

  if (output_prefix.empty()) {
    output_prefix = "gcmc_hybrid_output";
  }

  const bool moves_user = (t_nml->getKeywordStatus("--moves") == InputStatus::USER_SPECIFIED);
  const bool n_user = (t_nml->getKeywordStatus("-n") == InputStatus::USER_SPECIFIED);
  int n_moves = 0;
  if (moves_user && !n_user) {
    n_moves = t_nml->getIntValue("--moves");
  }
  else if (!moves_user && n_user) {
    n_moves = t_nml->getIntValue("-n");
  }
  else if (moves_user && n_user) {
    n_moves = t_nml->getIntValue("--moves");
  }
  else {
    n_moves = t_nml->getIntValue("--moves");
  }
  const double temperature = t_nml->getRealValue("--temp");
  const int n_ghosts = t_nml->getIntValue("--nghost");
  const double timestep = t_nml->getRealValue("--timestep");
  const int md_steps = t_nml->getIntValue("--md-steps");
  const double b_value = std::max(t_nml->getRealValue("--bvalue"), t_nml->getRealValue("-b"));
  const double mu_ex = t_nml->getRealValue("--mu-ex");
  const double standard_volume = t_nml->getRealValue("--standard-volume");

  const double mc_translation_raw = t_nml->getRealValue("--mc-translation");
  const double mc_rotation_raw = t_nml->getRealValue("--mc-rotation");
  const int mc_frequency_raw = t_nml->getIntValue("--mc-frequency");

  bool log_memory = t_nml->getBoolValue("--log-memory");
  const bool user_overrode_log_memory =
      (t_nml->getKeywordStatus("--log-memory") == InputStatus::USER_SPECIFIED);
  bool env_override_applied = false;
  if (!user_overrode_log_memory) {
    const char* env_log_memory = std::getenv("STORMM_LOG_MEMORY");
    bool env_value = false;
    if (readEnvBoolOverride(env_log_memory, &env_value)) {
      log_memory = env_value;
      env_override_applied = true;
    }
  }

  validateSimulationSettings(n_moves, temperature, timestep, md_steps, n_ghosts);
  validateMCSettings(mc_translation_raw, mc_rotation_raw, mc_frequency_raw);

  const double mc_translation = std::max(0.0, mc_translation_raw);
  const double mc_rotation = std::max(0.0, mc_rotation_raw);
  const int mc_frequency = std::max(0, mc_frequency_raw);

  const std::filesystem::path output_dir_path = ensureOutputDirectory(output_dir_cli);
  const std::filesystem::path output_base_path = combinePrefix(output_dir_path, output_prefix);
  std::optional<std::filesystem::path> final_pdb_path;
  if (!final_pdb_option.empty() && final_pdb_option != "none") {
    if (final_pdb_option == "auto" || final_pdb_option == "AUTO") {
      final_pdb_path = std::filesystem::path(output_base_path.string() + "_final.pdb");
    }
    else {
      std::filesystem::path candidate(final_pdb_option);
      if (!candidate.is_absolute()) {
        candidate = output_dir_path / candidate;
      }
      final_pdb_path = candidate;
    }
    if (final_pdb_path->parent_path() != std::filesystem::path()) {
      std::error_code dir_err;
      std::filesystem::create_directories(final_pdb_path->parent_path(), dir_err);
      if (dir_err) {
        rtErr("Unable to create directories for final PDB '" + final_pdb_path->string() +
              "': " + dir_err.message(), "gcmc_hybrid_runner");
      }
    }
  }

  std::filesystem::path constant_b_prefix_path;
  std::string constant_b_prefix_stem;
  const bool have_constant_b_prefix = !constant_b_pdb_prefix_cli.empty();
  if (have_constant_b_prefix) {
    constant_b_prefix_path = std::filesystem::path(constant_b_pdb_prefix_cli);
    if (!constant_b_prefix_path.is_absolute()) {
      constant_b_prefix_path = output_dir_path / constant_b_prefix_path;
    }
    constant_b_prefix_stem = constant_b_prefix_path.filename().string();
    if (constant_b_prefix_stem.empty()) {
      constant_b_prefix_stem = "constant_b";
    }
    const std::filesystem::path parent_dir = constant_b_prefix_path.parent_path();
    if (!parent_dir.empty()) {
      std::error_code dir_err;
      std::filesystem::create_directories(parent_dir, dir_err);
      if (dir_err) {
        rtErr("Unable to create directories for constant-B PDB prefix '" +
              constant_b_prefix_path.string() + "': " + dir_err.message(),
              "gcmc_hybrid_runner");
      }
    }
  }

  const bool use_adaptive_b = t_nml->getBoolValue("--adaptive-b");
  const int stage1_moves = t_nml->getIntValue("--stage1-moves");
  const int stage2_moves = t_nml->getIntValue("--stage2-moves");
  const int stage3_moves = t_nml->getIntValue("--stage3-moves");
  const double b_discovery = t_nml->getRealValue("--b-discovery");
  const double target_occupancy = t_nml->getRealValue("--target-occupancy");
  const double coarse_rate = t_nml->getRealValue("--coarse-rate");
  const double fine_rate = t_nml->getRealValue("--fine-rate");
  const double b_min = t_nml->getRealValue("--b-min");
  const double b_max = t_nml->getRealValue("--b-max");

  std::cout << "\n========================================\n";
  std::cout << "Hybrid GPU GCMC + MC Sampler\n";
  std::cout << "========================================\n";
  std::cout << "Protein topology:  " << protein_prmtop << "\n";
  std::cout << "Protein coords:    " << protein_inpcrd << "\n";
  std::cout << "Fragment topology: " << fragment_prmtop << "\n";
  std::cout << "Fragment coords:   " << fragment_inpcrd << "\n";
  std::cout << "GCMC cycles:       " << n_moves << "\n";
  std::cout << "Ghost templates:   " << n_ghosts << "\n";
  std::cout << "Temperature:       " << temperature << " K\n";
  std::cout << "MD steps / cycle:  " << md_steps << "\n";
  std::cout << "MC translation:    " << mc_translation << " Å\n";
  std::cout << "MC rotation:       " << mc_rotation << " deg\n";
  std::cout << "MC attempts/cycle: " << mc_frequency << "\n";
  std::cout << "Output directory:  " << output_dir_path << "\n";
  std::cout << "Output base path:  " << output_base_path << "\n";
  std::cout << "Adaptive B:        " << (use_adaptive_b ? "ON" : "OFF") << "\n";
  std::cout << "Log memory:        " << (log_memory ? "ON" : "OFF")
            << (env_override_applied ? " (STORMM_LOG_MEMORY)" : "") << "\n";
  if (final_pdb_path.has_value()) {
    std::cout << "Final PDB:         " << final_pdb_path->string()
              << (final_pdb_active_only ? " (active fragments only)" : "") << "\n";
  }
  else {
    std::cout << "Final PDB:         disabled\n";
  }
  if (have_constant_b_prefix) {
    std::cout << "Constant-B PDBs:   " << constant_b_prefix_path.parent_path() /
                 constant_b_prefix_stem
              << "_###.pdb (first 100 cycles)\n";
  }
  else {
    std::cout << "Constant-B PDBs:   disabled\n";
  }
  std::cout << "\n";

  try {
    AtomGraph combined_topology;
    PhaseSpace phase_space;
    std::vector<double> box_dims(6, 0.0);

    GhostMoleculeMetadata ghost_metadata =
        buildSystemWithGhosts(protein_prmtop, protein_inpcrd,
                              fragment_prmtop, fragment_inpcrd,
                              n_ghosts, &combined_topology, &phase_space, &box_dims);

    const int protein_atom_count = ghost_metadata.base_atom_count;
    std::cout << "Combined system:   " << combined_topology.getAtomCount() << " atoms, "
              << combined_topology.getMoleculeCount() << " molecules\n";
    std::cout << "Protein atoms:     " << protein_atom_count << "\n";
    std::cout << "Fragment templates:" << ghost_metadata.n_ghost_molecules << "\n";

    combined_topology.modifyAtomMobility(0, protein_atom_count, MobilitySetting::OFF);
    zeroProteinMasses(&combined_topology, protein_atom_count);
    zeroProteinVelocities(&phase_space, protein_atom_count);

    StaticExclusionMask exclusions(&combined_topology);

    Thermostat thermostat(combined_topology, ThermostatKind::LANGEVIN, temperature);
    thermostat.setTimeStep(timestep / 1000.0);
    thermostat.upload();

    const std::string output_base = output_base_path.string();
    const std::string ghost_file = output_base + "_ghosts.txt";
    const std::string log_file = output_base + "_gcmc.log";

    GCMCSystemSampler sampler(&combined_topology, &phase_space, &exclusions, &thermostat,
                              temperature, ghost_metadata, mu_ex, standard_volume,
                              b_value, 0.0, ImplicitSolventModel::NONE, "LIG",
                              ghost_file, log_file);

    if (mc_translation > 0.0) {
      sampler.enableTranslationMoves(mc_translation);
    }
    if (mc_rotation > 0.0) {
      sampler.enableRotationMoves(mc_rotation);
    }

    if (use_adaptive_b) {
      sampler.enableAdaptiveB(stage1_moves, stage2_moves, stage3_moves,
                              b_discovery, target_occupancy,
                              coarse_rate, fine_rate, b_min, b_max);
      std::cout << "Adaptive B schedule: [" << b_min << ", " << b_max << "] "
                << "discovery=" << b_discovery << " target="
                << target_occupancy << " (stage lengths: "
                << stage1_moves << ", " << stage2_moves << ", " << stage3_moves << ")\n";
    }
    else {
      std::cout << "Using constant Adams B = " << b_value << "\n";
    }

    sampler.invalidateEnergyCache();

    std::string occupancy_file = output_base + "_occupancy.dat";
    std::ofstream occ_out(occupancy_file);
    occ_out << "# Cycle  Active_Molecules\n";

    std::string memory_file = output_base + "_gpu_memory.dat";
    std::ofstream mem_out;
    if (log_memory) {
      mem_out.open(memory_file);
      if (!mem_out.is_open()) {
        rtErr("Cannot open memory file: " + memory_file, "gcmc_hybrid_runner");
      }
      mem_out << "# Cycle  GPU_Free_MB  GPU_Used_MB  GPU_Total_MB  RSS_MB  VMS_MB\n";
    }

    std::cout << "\nStarting simulation...\n";
    const int report_interval = std::max(50, n_moves / 10);
    int prev_report_moves = 0;
    int prev_report_accepts = 0;
    double total_gcmc_time = 0.0;
    double total_mc_time = 0.0;
    double total_md_time = 0.0;

    for (int cycle = 0; cycle < n_moves; cycle++) {
#ifdef STORMM_USE_HPC
      size_t free_bytes_before = 0UL;
      size_t total_bytes = 0UL;
      cudaMemGetInfo(&free_bytes_before, &total_bytes);
#endif
      const auto gcmc_start = clock::now();
      const bool accepted = sampler.runGCMCCycle();
      total_gcmc_time +=
          std::chrono::duration<double>(clock::now() - gcmc_start).count();

      const auto mc_start = clock::now();
      for (int mc_attempt = 0; mc_attempt < mc_frequency; mc_attempt++) {
        sampler.attemptMCMovesOnAllMolecules();
      }
      total_mc_time += std::chrono::duration<double>(clock::now() - mc_start).count();

      const auto md_start = clock::now();
      sampler.propagateSystem(md_steps);
      total_md_time += std::chrono::duration<double>(clock::now() - md_start).count();

      if (!use_adaptive_b && have_constant_b_prefix && cycle < 100) {
        const PhaseSpace snapshot_ps = sampler.exportCurrentPhaseSpace();
        std::ostringstream pdb_name;
        pdb_name << constant_b_prefix_stem << "_" << std::setw(3) << std::setfill('0')
                 << (cycle + 1) << ".pdb";
        const std::filesystem::path parent_dir = constant_b_prefix_path.parent_path();
        const std::filesystem::path snapshot_path =
            parent_dir.empty() ? std::filesystem::path(pdb_name.str())
                               : parent_dir / pdb_name.str();
        const std::vector<int> debug_atoms = sampler.getActiveAtomIndices();
        writePdbSnapshot(snapshot_path, combined_topology, snapshot_ps, debug_atoms,
                         "No active molecules");
      }

      if (log_memory) {
#ifdef STORMM_USE_HPC
        size_t free_bytes_after = 0UL;
        size_t total_bytes_after = 0UL;
        cudaError_t cuda_err = cudaMemGetInfo(&free_bytes_after, &total_bytes_after);
        double rss_mb = 0.0;
        double vms_mb = 0.0;
        std::ifstream status_file("/proc/self/status");
        if (status_file.is_open()) {
          std::string line;
          while (std::getline(status_file, line)) {
            if (line.rfind("VmRSS", 0) == 0) {
              size_t pos = line.find_first_of("0123456789");
              if (pos != std::string::npos) {
                rss_mb = std::stod(line.substr(pos)) / 1024.0;
              }
            }
            else if (line.rfind("VmSize", 0) == 0) {
              size_t pos = line.find_first_of("0123456789");
              if (pos != std::string::npos) {
                vms_mb = std::stod(line.substr(pos)) / 1024.0;
              }
            }
          }
          status_file.close();
        }
        if (cuda_err == cudaSuccess) {
          const double free_mb = static_cast<double>(free_bytes_after) / (1024.0 * 1024.0);
          const double total_mb = static_cast<double>(total_bytes_after) / (1024.0 * 1024.0);
          const double used_mb = total_mb - free_mb;
          mem_out << cycle << " "
                  << std::fixed << std::setprecision(2)
                  << free_mb << " " << used_mb << " " << total_mb << " "
                  << rss_mb << " " << vms_mb << "\n";
        }
        else {
          mem_out << cycle << " 0 0 0 0 0\n";
        }
#else
        mem_out << cycle << " 0 0 0 0 0\n";
#endif
        mem_out.flush();
      }

      occ_out << cycle << " " << sampler.getActiveCount() << "\n";
      occ_out.flush();

      std::cout << "Cycle " << std::setw(6) << cycle
                << "  Active=" << std::setw(4) << sampler.getActiveCount()
                << "  Move=" << (accepted ? "ACCEPT" : "REJECT") << "\n";

      const int completed_cycles = cycle + 1;
      if (report_interval > 0 &&
          (completed_cycles % report_interval == 0 || completed_cycles == n_moves)) {
        const GCMCStatistics& stats = sampler.getStatistics();
        const int block_moves = stats.n_moves - prev_report_moves;
        const int block_accepts = stats.n_accepted - prev_report_accepts;
        const double block_rate =
            (block_moves > 0) ?
            (static_cast<double>(block_accepts) * 100.0 /
             static_cast<double>(block_moves)) : 0.0;
        std::ostringstream block_stream;
        std::ostringstream overall_stream;
        block_stream << std::fixed << std::setprecision(2) << block_rate;
        overall_stream << std::fixed << std::setprecision(2) << stats.getAcceptanceRate();
        std::cout << "  Acceptance (last " << block_moves << "): "
                  << block_stream.str() << "%, overall: "
                  << overall_stream.str() << "%\n";
        prev_report_moves = stats.n_moves;
        prev_report_accepts = stats.n_accepted;
      }
    }

    const GCMCStatistics& final_stats = sampler.getStatistics();
    std::ostringstream overall_rate;
    std::ostringstream insert_rate;
    std::ostringstream delete_rate;
    overall_rate << std::fixed << std::setprecision(2) << final_stats.getAcceptanceRate();
    insert_rate << std::fixed << std::setprecision(2)
                << final_stats.getInsertionAcceptanceRate();
    delete_rate << std::fixed << std::setprecision(2)
                << final_stats.getDeletionAcceptanceRate();

    std::cout << "\nFinal statistics:\n";
    std::cout << "  Total moves:        " << final_stats.n_moves << "\n";
    std::cout << "  Accepted moves:     " << final_stats.n_accepted
              << " (" << overall_rate.str() << "%)\n";
    std::cout << "  Accepted insertions:" << final_stats.n_accepted_inserts
              << " (" << insert_rate.str() << "%)\n";
    std::cout << "  Accepted deletions: " << final_stats.n_accepted_deletes
              << " (" << delete_rate.str() << "%)\n";
    if (final_stats.n_explosions > 0 || final_stats.n_left_sphere > 0) {
      std::cout << "  Warnings:           ";
      if (final_stats.n_explosions > 0) {
        std::cout << final_stats.n_explosions << " integration failure(s)";
      }
      if (final_stats.n_left_sphere > 0) {
        if (final_stats.n_explosions > 0) {
          std::cout << ", ";
        }
        std::cout << final_stats.n_left_sphere << " molecule(s) left NCMC sphere";
      }
      std::cout << "\n";
    }
    std::cout << "  Active fragments:   " << sampler.getActiveCount()
              << " / " << sampler.getTotalMoleculeCount() << "\n";
    if (use_adaptive_b) {
      std::ostringstream current_b;
      current_b << std::fixed << std::setprecision(3) << sampler.getCurrentB();
      std::cout << "  Adaptive B stage:   " << describeAnnealingStage(sampler.getCurrentStage())
                << "\n";
      std::cout << "  Current B value:    " << current_b.str() << "\n";
      std::cout << "  Max fragments seen: " << sampler.getMaxFragments() << "\n";
    }
    else {
      std::ostringstream constant_b;
      constant_b << std::fixed << std::setprecision(3) << sampler.getCurrentB();
      std::cout << "  Constant B value:   " << constant_b.str() << "\n";
    }

    const double timing_denom = (n_moves > 0) ? static_cast<double>(n_moves) : 1.0;
    std::cout << "  Avg timing per cycle (s): GCMC="
              << (total_gcmc_time / timing_denom)
              << "  MC=" << (total_mc_time / timing_denom)
              << "  MD=" << (total_md_time / timing_denom) << "\n";

    if (final_pdb_path.has_value()) {
      const PhaseSpace final_ps = sampler.exportCurrentPhaseSpace();
      std::vector<int> final_atoms;
      std::string empty_message = "No atoms selected";
      if (final_pdb_active_only) {
        final_atoms.reserve(static_cast<size_t>(protein_atom_count));
        for (int i = 0; i < protein_atom_count; i++) {
          final_atoms.push_back(i);
        }
        const std::vector<int> active_atoms = sampler.getActiveAtomIndices();
        final_atoms.insert(final_atoms.end(), active_atoms.begin(), active_atoms.end());
        empty_message = active_atoms.empty() ?
                        "Protein only (no active fragments)" :
                        "Protein plus active fragments";
      }
      else {
        const int natoms = combined_topology.getAtomCount();
        final_atoms.resize(natoms);
        std::iota(final_atoms.begin(), final_atoms.end(), 0);
      }
      writePdbSnapshot(*final_pdb_path, combined_topology, final_ps,
                       final_atoms, empty_message);
      std::cout << "Final PDB snapshot written to " << final_pdb_path->string() << "\n";
    }

    std::cout << "\nSimulation complete.\n";
  }
  catch (std::exception &e) {
    rtErr("Hybrid GCMC encountered an error: " + std::string(e.what()),
          "gcmc_hybrid_runner");
  }

  return 0;
}
