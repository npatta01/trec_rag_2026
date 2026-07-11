import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.EnglishPossessiveFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

/** Read-only localhost service for the frozen Anserini default English chain. */
public final class AnalyzerServer {
  private static final int MAX_BODY_BYTES = 1024 * 1024;
  private static final CharArraySet STOPWORDS = EnglishAnalyzer.ENGLISH_STOP_WORDS_SET;
  private static final Analyzer ANALYZER = new Analyzer() {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      StandardTokenizer source = new StandardTokenizer();
      TokenStream stream = new EnglishPossessiveFilter(source);
      stream = new LowerCaseFilter(stream);
      stream = new StopFilter(stream, STOPWORDS);
      stream = new PorterStemFilter(stream);
      return new TokenStreamComponents(source, stream);
    }
  };
  private static final String INDEX_ID = System.getenv().getOrDefault(
      "ANALYZER_INDEX_ID", "hosted_climbmix_unknown_revision");
  private static final String FINGERPRINT_JSON = fingerprintJson();

  private AnalyzerServer() {}

  public static void main(String[] args) throws Exception {
    int port = Integer.parseInt(System.getenv().getOrDefault("ANALYZER_PORT", "18081"));
    HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
    server.createContext("/health", AnalyzerServer::health);
    server.createContext("/analyze", AnalyzerServer::analyze);
    server.setExecutor(Executors.newFixedThreadPool(2));
    server.start();
    System.out.println("Lucene analyzer server ready on port " + port);
  }

  private static void health(HttpExchange exchange) throws IOException {
    if (!exchange.getRequestMethod().equals("GET")) {
      respond(exchange, 405, "{\"error\":\"method not allowed\"}");
      return;
    }
    respond(exchange, 200, "{\"status\":\"ok\",\"fingerprint\":"
        + FINGERPRINT_JSON + "}");
  }

  private static void analyze(HttpExchange exchange) throws IOException {
    if (!exchange.getRequestMethod().equals("POST")) {
      respond(exchange, 405, "{\"error\":\"method not allowed\"}");
      return;
    }
    byte[] body = exchange.getRequestBody().readNBytes(MAX_BODY_BYTES + 1);
    if (body.length > MAX_BODY_BYTES) {
      respond(exchange, 413, "{\"error\":\"request body too large\"}");
      return;
    }
    String text = new String(body, StandardCharsets.UTF_8);
    List<String> tokens;
    try {
      tokens = analyzeText(text);
    } catch (Exception error) {
      respond(exchange, 500, "{\"error\":\"analysis failed\"}");
      return;
    }
    StringBuilder json = new StringBuilder("{\"tokens\":[");
    for (int index = 0; index < tokens.size(); index++) {
      if (index > 0) json.append(',');
      json.append(jsonString(tokens.get(index)));
    }
    json.append("],\"fingerprint\":").append(FINGERPRINT_JSON).append('}');
    respond(exchange, 200, json.toString());
  }

  private static List<String> analyzeText(String text) throws Exception {
    List<String> tokens = new ArrayList<>();
    try (TokenStream stream = ANALYZER.tokenStream("contents", text)) {
      CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
      stream.reset();
      while (stream.incrementToken()) tokens.add(term.toString());
      stream.end();
    }
    return tokens;
  }

  private static String fingerprintJson() {
    List<String> stopwords = new ArrayList<>();
    for (Object value : STOPWORDS) {
      if (value instanceof char[]) stopwords.add(new String((char[]) value));
      else stopwords.add(value.toString());
    }
    Collections.sort(stopwords);
    String stopwordHash = sha256(String.join("\n", stopwords));
    return "{"
        + "\"contract_version\":\"lucene_default_english_v1\","
        + "\"implementation\":\"local_lucene_reference_server_v1\","
        + "\"lucene_version\":" + jsonString(Version.LATEST.toString()) + ","
        + "\"analyzer_class\":\"io.anserini.analysis.DefaultEnglishAnalyzer chain\","
        + "\"tokenizer\":\"org.apache.lucene.analysis.standard.StandardTokenizer\","
        + "\"filters\":["
        + "\"EnglishPossessiveFilter\",\"LowerCaseFilter\","
        + "\"StopFilter(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET)\","
        + "\"PorterStemFilter\"],"
        + "\"stopword_sha256\":" + jsonString(stopwordHash) + ","
        + "\"unicode_version\":\"Lucene-10.4.0-StandardTokenizer-UAX29\","
        + "\"index_id\":" + jsonString(INDEX_ID)
        + "}";
  }

  private static String sha256(String value) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      byte[] bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8));
      StringBuilder result = new StringBuilder();
      for (byte item : bytes) result.append(String.format("%02x", item));
      return result.toString();
    } catch (Exception error) {
      throw new IllegalStateException(error);
    }
  }

  private static String jsonString(String value) {
    StringBuilder result = new StringBuilder("\"");
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      switch (character) {
        case '\\': result.append("\\\\"); break;
        case '"': result.append("\\\""); break;
        case '\n': result.append("\\n"); break;
        case '\r': result.append("\\r"); break;
        case '\t': result.append("\\t"); break;
        default:
          if (character < 0x20) result.append(String.format("\\u%04x", (int) character));
          else result.append(character);
      }
    }
    return result.append('"').toString();
  }

  private static void respond(HttpExchange exchange, int status, String body)
      throws IOException {
    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
    exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
    exchange.getResponseHeaders().set("Cache-Control", "no-store");
    exchange.sendResponseHeaders(status, bytes.length);
    try (OutputStream output = exchange.getResponseBody()) {
      output.write(bytes);
    }
  }
}
