package assignment3;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.google.common.collect.Iterables;

import csd1054.WordCount;

public class InvertIndexFrequency extends Configured implements Tool {
	
	private static final Logger LOG = Logger.getLogger(WordCount.class);

	@Override
	public int run(String[] args) throws Exception {
		
		Job job = Job.getInstance(getConf(), "invertindex");
		Job idjob = Job.getInstance(getConf(), "idjob");
		int j = 0;
		
		job.setJarByClass(this.getClass());
		
		for(j = 0; j < args.length - 1; j++){
			FileInputFormat.addInputPath(job, new Path(args[j]));
		}
		FileOutputFormat.setOutputPath(job, new Path(args[j]));
		
		job.setMapperClass(Map.class);
	    job.setCombinerClass(Combiner.class);
	    job.setReducerClass(Reduce.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(Text.class);
		
	    job.waitForCompletion(true);
	    
	    idjob.setJarByClass(this.getClass());
		
	    FileInputFormat.addInputPath(idjob, new Path("ex3/part-r-00000"));
		FileOutputFormat.setOutputPath(idjob, new Path("finalOutput2"));
		
		idjob.setMapperClass(SecondMap.class);
		idjob.setCombinerClass(SecondReduce.class);
		idjob.setReducerClass(SecondReduce.class);
		idjob.setOutputKeyClass(Text.class);
		idjob.setOutputValueClass(Text.class);
	    
		return idjob.waitForCompletion(true) ? 1 : 0;
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new InvertIndexFrequency(), args);
		System.exit(res);
	}
	
	private static class Map extends Mapper<LongWritable, Text, Text, Text>{
		 
	    private boolean caseSensitive = false;
	    private String input;
	    private Set<String> patternsToSkip = new HashSet<String>();
	    private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");
	    private final String regex = "[a-zA-z]+";
		private ArrayList<String> stopwords = new ArrayList<String>();
		
		protected void setup(Mapper.Context context)
			        throws IOException,
			        InterruptedException {
		if (context.getInputSplit() instanceof FileSplit) {
	        this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
	      } else {
	        this.input = context.getInputSplit().toString();
	      }
	      Configuration config = context.getConfiguration();
	      this.caseSensitive = config.getBoolean("wordcount.case.sensitive", false);
	      if (config.getBoolean("invertIndex.skip.patterns", false)) {
	        URI[] localPaths = context.getCacheFiles();
	        parseSkipFile(localPaths[0]);
	      }
	      setupStopWords(stopwords, context);
	    }

		private void setupStopWords(ArrayList<String> stopwords, Context context) {
	    	Path stopwordspath = new Path("stopwords.csv");
	    	String input = new String();
	    	try{
	    		FileSystem fs = FileSystem.get(context.getConfiguration());
	    		if(fs.exists(stopwordspath)){
	    			FSDataInputStream in = fs.open(stopwordspath);
	    			BufferedReader buf = new BufferedReader(new InputStreamReader(in));
	    			while((input = buf.readLine()) != null){
	    				stopwords.add(input.substring(2));
	    			}
		    		in.close();
		    		
	    		}
	    		else{
	    			LOG.info("Stopwords file not found!");
	    		}
	    		fs.close();
	    	}
	    	catch(Exception e){
	    		System.err.println("Caught exception while parsing stopwords file: " + StringUtils.stringifyException(e));
	    	}
		}

		private void parseSkipFile(URI patternsURI) {
	      LOG.info("Added file to the distributed cache: " + patternsURI);
	      try {
	        BufferedReader fis = new BufferedReader(new FileReader(new File(patternsURI.getPath()).getName()));
	        String pattern;
	        while ((pattern = fis.readLine()) != null) {
	          patternsToSkip.add(pattern);
	        }
	      } catch (IOException ioe) {
	        System.err.println("Caught exception while parsing the cached file '"
	            + patternsURI + "' : " + StringUtils.stringifyException(ioe));
	      }
	    }

	    public void map(LongWritable offset, Text lineText, Context context)
	        throws IOException, InterruptedException {
	      String line = lineText.toString();
	      String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
	      if (!caseSensitive) {
	        line = line.toLowerCase();
	      }
	      Text currentWord = new Text();
	      for (String word : WORD_BOUNDARY.split(line)) {
	        if (word.isEmpty() || !word.matches(regex) || stopwords.contains(word)) {
	            continue;
	        }
	            currentWord = new Text(word);
	            context.write(currentWord,new Text(fileName));
	        }             
	    }
	}
	
	private static class Combiner extends Reducer<Text, Text, Text, Text>{
		
		 @Override
		    public void reduce(Text word, Iterable<Text> docs, Context context)
		        throws IOException, InterruptedException {
		     
			 String previous = new String();
			 String current = new String();
			 StringBuilder str = new StringBuilder();
		     
			 int counter = 1;
			 previous = Iterables.get(docs, 0).toString();
			 
		     for (Text doc : docs) {
		    	 current = doc.toString();
		    	 if(current.equals(previous)){
		    		 counter++;
		    		 continue;
		    	 }
		    	 str.append(previous + " #"+counter +", ");
		    	 previous = new String(doc.toString());
		    	 counter = 1;
		    	 LOG.info(str);
		      }
		     str.append(previous + " #"+counter +", ");
		     context.write(word, new Text(str.toString()));
		    }
	}
	
	
	private static class Reduce extends Reducer<Text, Text, Text, Text>{
		
		
		 @Override
		    public void reduce(Text word, Iterable<Text> docs, Context context)
		        throws IOException, InterruptedException {
		      StringBuilder str = new StringBuilder();
		      
		      for (Text doc : docs) {
		    	  str.append(doc);
		      }
		      context.write(word, new Text(str.toString()));
		    }
	}
	
	
	private static class SecondMap extends Mapper<LongWritable, Text, Text, Text>{
		
	    private boolean caseSensitive = false;
	    private long numRecords = 0;
	    private String input;
	    private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");
		
		protected void setup(Mapper.Context context)
		        throws IOException,
		        InterruptedException {
			if (context.getInputSplit() instanceof FileSplit) {
				this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
			} else {
				this.input = context.getInputSplit().toString();
			}
			Configuration config = context.getConfiguration();
			
			this.caseSensitive = config.getBoolean("wordcount.case.sensitive", false);
			config.setInt("idCounter", 0);
		}
		
		public void map(LongWritable offset, Text lineText, Context context) throws IOException, InterruptedException {
		      String line = lineText.toString();
		      int counter;
		      if (!caseSensitive) {
		        line = line.toLowerCase();
		      }
		      Text key = new Text();
		      Text values = new Text();
		      String[] splitLine = new String[2];
		      //split line in tab to get the word and the documents
		      splitLine = line.split("\t");
		      key = new Text(splitLine[0]);
		      values = new Text(splitLine[1]);
		      counter = context.getConfiguration().getInt("idCounter", 0);
		      counter++;
		      context.getConfiguration().setInt("idCounter", counter);	         
		      context.write(key,new Text(values + " " + counter));             
		    }
		
		
	}
	
	
	
	private static class SecondReduce extends Reducer<Text, Text, Text, Text>{
		
		
		 @Override
		    public void reduce(Text word, Iterable<Text> docs, Context context)
		        throws IOException, InterruptedException {
		      StringBuilder str = new StringBuilder();

		      for (Text doc : docs) {
		    	  str.append(doc);
		      }
		      context.write(word, new Text(str.toString()));
		    }
	}
}
