function disquote(str){
    if ((str.startsWith('"') && str.endsWith('"')) ||
        (str.startsWith("'") && str.endsWith("'"))){
        return str.substring(1, str.length-1);
    }
    else {
        return str;
    }
}

function createTerminal(cstnode){
    let node_val = cstnode.node;
    switch(node_val.toLowerCase()) {
        case 'specific': return disquote(cstnode.subtrees[0].node);
        case 'characterrange': return '[' + disquote(cstnode.subtrees[0].node) + '-' + disquote(cstnode.subtrees[1].node) + ']';
        case 'empty': return '';
        case 'word': return '\\w';
        case 'any': return '.';
        case 'digit': return '\\d';
        case 'space': return '\\s';
        case 'notword': return '\\W';
        case 'notdigit': return '\\D';
        case 'notspace': return '\\S';
        default: throw Error('Unknown terminal type: ' + node_val);         
    }
}   

function createConcat(cstnode){
    return cstnode.reduce( (obj,el) => {
        return obj + createRegex(el);
    }, '');
}

// Creates a string of regexes from the CST
// CST is now represented as a json dictionary with a top node
// Each node has a "node" value,  a grammar "type" and a subtrees list
// So this function should expect to get a node, not a list
function createRegex(cstnode){
    let node_val = cstnode.node;
    switch(node_val.toLowerCase()) {
        case 'terminal': return createTerminal(cstnode.subtrees[0]);
        case 'startofline': return '(^' + createRegex(cstnode.subtrees[0]) + ')';
        case 'endofline': return '($' + createRegex(cstnode.subtrees[0]) + ')';
        case 'plus': return '(' + createRegex(cstnode.subtrees[0]) + '+)';
        case 'star': return '(' + createRegex(cstnode.subtrees[0]) + '*)';
        case 'or': return '(' + createRegex(cstnode.subtrees[0]) + '|' + createRegex(cstnode.subtrees[1]) + ')';
        case 'concat': return '(' + createConcat(cstnode.subtrees[0]) + ')';
        default: throw Error('Unknown regex form: ' + node_val);  
    }  
}

export function createRegexes(csts){
    return csts.sentences.reduce( (obj, sent) => {
        if (sent.cst){
            return obj + createRegex(sent.cst) + '\n';    
        }
        else {
            return obj;
        }
    }, '');
}
